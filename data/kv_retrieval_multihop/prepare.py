"""
kv_retrieval_multihop synthetic dataset generator (token-level construction)

要点：
- 直接以整数 token 构造，不走文本编码。
- 特殊 token 标注 DB/QA 起止、K->V 的箭头、DB 内分隔符（来自一个小的可选集合，插入 1-3 个）、QA 项之间的唯一分隔符、括号/逗号、答案起始、T/F、PAD、EOT。
- 多跳链：每条链长度为 HOPS，每一跳的 Key 是上一跳的 Value（均为 token 序列，长度 3~6）。
- DB 将多条链的所有 K->V pair 混排，并插入若干无关的干扰 pair；不同 pair 之间用 1~3 个 DB 分隔 token 隔开。
- QA 形式为 (K0, V_goal) ANS_START LABEL，再以 QA 专用分隔符分隔下一条。正/负例尽量 50%。
- 所有 token 使用三位数小整数，便于阅读对齐；.txt 直接打印空格分隔的整数序列。

环境变量：
- TRAIN_EPISODES, VAL_EPISODES, SEED, BLOCK_SIZE, WRITE_TXT, WORKERS, OUT_DIR, HOPS, KV_VOCAB, SEP_VOCAB
"""

import os
import random
import pickle
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
from tqdm.auto import tqdm


# -----------------------
# Special token assignment
# -----------------------
# Keep tokens as 3-digit integers (>= 100) for readability, as requested.
DB_START        = 170
QA_START        = 171
ARROW           = 172  # represents '->' between K and V
VOCAB_START     = 200  # K/V tokens start from here
VOCAB_SIZE      = 300
PAIR_SEP_START  = 800
PAIR_SEP_SIZE   = 50   # DB separators (50 options)
COMMA           = 182
ANS_START       = 183  # marks the start of an answer; the next token is the T/F label
ANS_TRUE        = 184
ANS_FALSE       = 185
PAD             = 186
EOT             = 199  # episode terminator appended to reach block_size


def _sample_seq(rng: random.Random, kv_pool: List[int], min_len: int = 3, max_len: int = 6) -> Tuple[int, ...]:
    L = rng.randint(min_len, max_len)
    return tuple(rng.choices(kv_pool, k=L))


def build_chain_tokens(
    rng: random.Random, kv_pool: List[int], hops: int,
) -> Tuple[List[Tuple[Tuple[int, ...], Tuple[int, ...]]], Tuple[int, ...], Tuple[int, ...]]:
    """构造一条长度 hops 的链，满足 K_{t+1} == V_t，K/V 为长度 3~6 的 token 序列。

    返回：
    - pairs: [(K_seq, V_seq), ...] 共 hops 个
    - k0_seq: 初始 Key 序列
    - v_last_seq: 最终 Value 序列
    """
    nodes: List[Tuple[int, ...]] = []
    used = set()
    while len(nodes) < hops + 1:
        s = _sample_seq(rng, kv_pool)
        if s not in used:
            nodes.append(s)
            used.add(s)
    k0 = nodes[0]
    v_last = nodes[-1]
    pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for t in range(hops):
        pairs.append((nodes[t], nodes[t+1]))
    return pairs, k0, v_last


def assemble_db(
    rng: random.Random,
    kv_pool: List[int],
    sep_pool: List[int],
    hops: int,
    target_tokens: int,
    db_token_budget: int,
    qa_budget: int,
    min_chains: int = 2,
    max_chains: int = 12,
) -> Tuple[List[int], List[Tuple[Tuple[int, ...], Tuple[int, ...]]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]], Dict[Tuple[int, ...], Tuple[int, ...]]]:
    """Assemble the DB token sequence and return:
    - db_ids: token list beginning with DB_START and KV pairs separated by 1~3 random DB separators
    - all_pairs: list of all (K_seq, V_seq) included (from chains + distractors)
    - chain_pairs: list of (K_seq, V_seq) that belong to the chains (subset of all_pairs)
    - chain_start_to_goal: map K0_seq -> V_goal_seq (只保留完整放入 DB 的链)

    DB token 预算为 `db_token_budget`。
    """
    db_ids: List[int] = [DB_START]
    all_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    chain_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    chain_start_to_goal: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    # 动态确定链条数量，尽量保证完整链入库。
    # 同时考虑 QA 预算，确保有足够多的不同起点 K0 来生成互不相同的 QA。
    avg_seq = 4.5
    avg_sep = 2.0  # 1~3 的均值
    avg_pair_tokens = int(round(avg_seq + 1 + avg_seq + avg_sep))  # K + ARROW + V + SEP
    # 估计 DB 能放入的链数量（保守估计）
    est_chains_db = max(min_chains, min(max_chains, max(1, db_token_budget // max(1, (avg_pair_tokens * hops)))))
    # 估计每个 QA 项平均需要的 token（'(' K ',' V ')' ANS_START LABEL 大约为以下量级）
    avg_qa_tokens = int(round(2 * avg_seq + 5))
    # 期望的 QA 条目数（受 qa_budget 限制）
    expected_qa_items = max(1, (qa_budget + avg_qa_tokens - 1) // avg_qa_tokens)
    # 希望的最小链数量应至少能提供 expected_qa_items 个不同的 K0（加一点余量）
    desired_chains = min(max_chains, max(min_chains, expected_qa_items + 1))
    # 最终决策基于 DB 能放的链和 QA 需求两者的上限
    est_chains = max(est_chains_db, desired_chains)

    constructed = 0
    attempts_chains = 0
    # 尝试构造链条，优先保证 K0 唯一以支撑互不相同的 QA
    while constructed < est_chains and attempts_chains < 5000:
        pairs, k0, v_last = build_chain_tokens(rng, kv_pool, hops)
        attempts_chains += 1
        # 如果已有相同起点 K0，则跳过以避免重复 QA 起点
        if k0 in chain_start_to_goal:
            continue
        # 预估该链 token
        est_tokens = 0
        for kseq, vseq in pairs:
            est_tokens += len(kseq) + 1 + len(vseq) + rng.randint(1, 3)
        # 如果放不下整个链，就结束尝试，保持已放入链完整性
        if len(db_ids) + est_tokens > db_token_budget:
            break
        # 放入链条
        for kseq, vseq in pairs:
            db_ids.extend(list(kseq))
            db_ids.append(ARROW)
            db_ids.extend(list(vseq))
            # 1~3 个 DB 分隔 token
            for _ in range(rng.randint(1, 3)):
                db_ids.append(rng.choice(sep_pool))
            chain_pairs.append((kseq, vseq))
            all_pairs.append((kseq, vseq))
        # 记录链的起点到最终值的映射，确保 K0 唯一
        chain_start_to_goal[k0] = v_last
        constructed += 1

    # 如果在首次构造后仍然没有足够的不同起点以支撑 QA，尝试额外插入更多完整链（受 db_token_budget 限制）
    extra_attempts = 0
    while len(chain_start_to_goal) < desired_chains and extra_attempts < 2000 and len(db_ids) < db_token_budget - 8:
        pairs, k0, v_last = build_chain_tokens(rng, kv_pool, hops)
        extra_attempts += 1
        if k0 in chain_start_to_goal:
            continue
        # 估算该链需要的 token
        est_tokens = 0
        for kseq, vseq in pairs:
            est_tokens += len(kseq) + 1 + len(vseq) + rng.randint(1, 3)
        if len(db_ids) + est_tokens > db_token_budget:
            break
        # 放入链条
        for kseq, vseq in pairs:
            db_ids.extend(list(kseq))
            db_ids.append(ARROW)
            db_ids.extend(list(vseq))
            for _ in range(rng.randint(1, 3)):
                db_ids.append(rng.choice(sep_pool))
            chain_pairs.append((kseq, vseq))
            all_pairs.append((kseq, vseq))
        chain_start_to_goal[k0] = v_last
        constructed += 1

    # 插入干扰 pair，直到接近预算
    existing = set(chain_pairs)
    attempts = 0
    while attempts < 2000 and len(db_ids) < db_token_budget - 8:  # 留小余量
        kseq = _sample_seq(rng, kv_pool)
        vseq = _sample_seq(rng, kv_pool)
        if kseq == vseq:
            attempts += 1
            continue
        pair = (kseq, vseq)
        if pair in existing:
            attempts += 1
            continue
        est = len(kseq) + 1 + len(vseq) + rng.randint(1, 3)
        if len(db_ids) + est > db_token_budget:
            break
        # 写入
        db_ids.extend(list(kseq))
        db_ids.append(ARROW)
        db_ids.extend(list(vseq))
        for _ in range(rng.randint(1, 3)):
            db_ids.append(rng.choice(sep_pool))
        all_pairs.append(pair)
        existing.add(pair)
        attempts += 1

    return db_ids, all_pairs, chain_pairs, chain_start_to_goal


def assemble_qa(
    rng: random.Random,
    chain_start_to_goal: Dict[Tuple[int, ...], Tuple[int, ...]],
    qa_token_budget: int,
    pos_ratio: float = 0.5,
) -> List[int]:
    """Assemble QA section token sequence.

    QA format: K0_seq COMMA V_goal_seq ANS_START (T/F)
    正负尽量 50%，并受预算约束。
    """
    qa_ids: List[int] = [QA_START]
    starts = list(chain_start_to_goal.keys())
    rng.shuffle(starts)
    pos_first = rng.random() < pos_ratio
    idx = 0
    while idx < len(starts):
        k0 = starts[idx]
        v_goal_true = chain_start_to_goal[k0]
        make_pos = pos_first if (idx % 2 == 0) else (not pos_first)
        # 决定标签并挑选目标
        if make_pos:
            v_goal = v_goal_true
            label = ANS_TRUE
        else:
            finals = [v for v in chain_start_to_goal.values() if v != v_goal_true]
            if not finals:
                v_goal = v_goal_true
                label = ANS_TRUE
            else:
                v_goal = rng.choice(finals)
                label = ANS_FALSE
        # 估计长度
        need = 1 + len(k0) + 1 + len(v_goal) + 1 + 1 + 1  # '(' K ',' V ')' ANS_START LABEL
        if len(qa_ids) + need > qa_token_budget:
            break
        qa_ids.extend(list(k0))
        qa_ids.append(COMMA)
        qa_ids.extend(list(v_goal))
        qa_ids.append(ANS_START)
        qa_ids.append(label)
        idx += 1

    return qa_ids


def build_episode(
    rng: random.Random,
    hops: int,
    kv_pool: List[int],
    sep_pool: List[int],
    target_tokens: int,
) -> List[int]:
    """Build one episode as token IDs of length == target_tokens (without final EOT).

    Composition: DB section (~2/3 budget), QA section (~1/3), then PAD to exact length.
    """
    # Split budget according to DB:QA = 2N:1 where N = hops
    # DB fraction = (2*hops) / (2*hops + 1)
    if hops <= 0:
        hops = 1
    db_budget = int(target_tokens * (2 * hops) / (2 * hops + 1))
    qa_budget = target_tokens - db_budget

    # DB
    db_ids, all_pairs, chain_pairs, start_to_goal = assemble_db(
        rng, kv_pool, sep_pool, hops, target_tokens, db_budget, qa_budget
    )

    # QA: choose from starts and final values present in chains
    qa_ids = assemble_qa(rng, start_to_goal, qa_budget)

    ids = db_ids + qa_ids

    # Pad to exact length with PAD
    if len(ids) < target_tokens:
        ids.extend([PAD] * (target_tokens - len(ids)))
    else:
        # If we slightly overflow due to rounding, truncate to target length
        ids = ids[:target_tokens]

    assert len(ids) == target_tokens
    return ids


def write_split(
    out_path: str,
    episodes: int,
    seed: int,
    block_size: int,
    write_txt: bool,
    workers: int,
    hops: int,
    kv_start: int,
    kv_vocab: int,
    sep_start: int,
    sep_vocab: int,
):
    """Write a dataset split to .bin (uint16) and optional .txt (space-separated ints)."""
    text_path = out_path.replace('.bin', '.txt')
    with open(out_path, 'wb') as f, (open(text_path, 'w', encoding='utf-8') if write_txt else nullcontext()) as ftxt:
        total_tokens = 0
        target_tokens = block_size - 1

        # Build pools
        # K/V pool: choose three-digit tokens 100..(100+kv_vocab-1)
        kv_pool = list(range(kv_start, kv_start + kv_vocab))
        sep_pool = list(range(sep_start, sep_start + sep_vocab))

        if workers is None or workers <= 1:
            rng = random.Random(seed)
            pbar = tqdm(total=episodes, desc=os.path.basename(out_path), leave=False)
            for _ in range(episodes):
                ids = build_episode(rng, hops, kv_pool, sep_pool, target_tokens)
                ids_eot = ids + [EOT]
                assert len(ids_eot) == block_size
                arr = np.array(ids_eot, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(arr)
                if write_txt and ftxt is not None:
                    ftxt.write(' '.join(str(t) for t in ids))
                    ftxt.write(f"\n{EOT}\n\n")
                pbar.update(1)
            pbar.close()
        else:
            # multiprocessing workers: create per-episode seeds for determinism
            def_args = [(seed, ep_index, hops, kv_pool, sep_pool, target_tokens) for ep_index in range(episodes)]

            adaptive_chunksize = max(1, min(256, (episodes // (workers * 8)) or 1))
            with mp.get_context('spawn').Pool(processes=workers) as pool:
                for ids in tqdm(pool.imap(_build_episode_worker, def_args, chunksize=adaptive_chunksize), total=episodes, desc=os.path.basename(out_path), leave=False):
                    ids_eot = ids + [EOT]
                    assert len(ids_eot) == block_size
                    arr = np.array(ids_eot, dtype=np.uint16)
                    arr.tofile(f)
                    total_tokens += len(arr)
                    if write_txt and ftxt is not None:
                        ftxt.write(' '.join(str(t) for t in ids))
                        ftxt.write(f"\n{EOT}\n\n")


def _build_episode_worker(args_tuple) -> List[int]:
    seed, ep_index, hops, kv_pool, sep_pool, target_tokens = args_tuple
    r = random.Random(seed + ep_index)
    return build_episode(r, hops, kv_pool, sep_pool, target_tokens)


def main():
    TRAIN_EPISODES = int(os.environ.get('TRAIN_EPISODES', 200_000))
    VAL_EPISODES = int(os.environ.get('VAL_EPISODES', 20_000))
    SEED = int(os.environ.get('SEED', 1337))
    BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', 1024))
    WRITE_TXT = int(os.environ.get('WRITE_TXT', 1)) > 0
    default_workers = 8
    try:
        cpu_cnt = mp.cpu_count()
    except Exception:
        cpu_cnt = default_workers
    WORKERS = int(os.environ.get('WORKERS', default_workers))
    if WORKERS < 1:
        WORKERS = 1
    WORKERS = min(WORKERS, cpu_cnt) if isinstance(cpu_cnt, int) and cpu_cnt > 0 else WORKERS

    HOPS = int(os.environ.get('HOPS', 4))
    KV_VOCAB = int(os.environ.get('KV_VOCAB', VOCAB_SIZE))
    SEP_VOCAB = int(os.environ.get('SEP_VOCAB', PAIR_SEP_SIZE))

    dstdir = os.environ.get('OUT_DIR', os.path.dirname(__file__))
    os.makedirs(dstdir, exist_ok=True)
    train_path = os.path.join(dstdir, 'train.bin')
    val_path = os.path.join(dstdir, 'val.bin')

    print("Generating kv_retrieval_multihop dataset...")
    print(f"train episodes: {TRAIN_EPISODES}, val episodes: {VAL_EPISODES}")
    print(f"episode composition: DB:QA = 2*HOPS:1, hops={HOPS}")
    print(f"block_size: {BLOCK_SIZE}, seed: {SEED}, write_txt: {WRITE_TXT}, workers: {WORKERS}")

    # Ensure train and val use the same separator token start defined by PAIR_SEP_START
    write_split(train_path, TRAIN_EPISODES, SEED, BLOCK_SIZE, WRITE_TXT, WORKERS, HOPS, VOCAB_START, KV_VOCAB, PAIR_SEP_START, SEP_VOCAB)
    write_split(val_path, VAL_EPISODES, SEED + 1, BLOCK_SIZE, WRITE_TXT, WORKERS, HOPS, VOCAB_START, KV_VOCAB, PAIR_SEP_START, SEP_VOCAB)

    meta = {
        # keep vocab small and contiguous; ensure it covers all tokens we may emit
        'vocab_size': max(VOCAB_START + VOCAB_SIZE, PAIR_SEP_START + SEP_VOCAB, ANS_FALSE, EOT) + 1,
        'block_size': BLOCK_SIZE,
        'desc': 'kv_retrieval_multihop synthetic dataset (token-level). DB: K->V pairs, QA: (K,V_goal) -> T/F.',
        'format_version': 2,
        # for convenience, store special token ids
        'special_tokens': {
            'DB_START': DB_START,
            'QA_START': QA_START,
            'ARROW': ARROW,
            'COMMA': COMMA,
            'ANS_START': ANS_START,
            'ANS_TRUE': ANS_TRUE,
            'ANS_FALSE': ANS_FALSE,
            'PAD': PAD,
            'EOT': EOT,
        },
        'hops': HOPS,
        'kv_vocab': KV_VOCAB,
        'sep_vocab': SEP_VOCAB,
    }
    with open(os.path.join(dstdir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    main()
