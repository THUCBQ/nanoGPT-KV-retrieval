"""
kv_retrieval_multihop synthetic dataset generator (text-first version).

Updates:
- First build human-readable DB/QA text (similar to kv_retrieval), then encode to token ids.
- Keeps dataset organization (DB/QA budgeting, multihop chain construction, block sizing) and
    still writes train/val .bin, optional .txt, and meta.pkl.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import random
import string
import sys
from contextlib import nullcontext
from typing import Iterable, List, Tuple

import numpy as np
import tiktoken
from tqdm.auto import tqdm


# Defaults and limits
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 7
UINT16_MAX_SAFE = 65535  # allow full uint16 range
DEFAULT_KV_VOCAB = 10  # kept for env symmetry; now drives unique string space only
DEFAULT_SEP_VOCAB = 1000  # unused in text mode, placeholder for env compatibility
ALPHABET = string.ascii_lowercase + string.digits

_ENC = None  # global encoder for workers


def _env_int(name: str, default: int) -> int:
    """Parse int environment variable with fallback."""
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _check_token_range(tokens: Iterable[int]) -> None:
    for t in tokens:
        if t < 0 or t > UINT16_MAX_SAFE:
            raise ValueError(f"token {t} out of allowed range 0..{UINT16_MAX_SAFE}")


def _worker_init() -> None:
    global _ENC
    _ENC = tiktoken.get_encoding('gpt2')


def _sample_seq(rng: random.Random, seen: set = None) -> str:
    """Sample a random string (length MIN..MAX) not present in `seen`."""
    max_attempts = 1000
    seq = ''
    for _ in range(max_attempts):
        length = rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
        seq = ''.join(rng.choices(ALPHABET, k=length))
        if seen is None or seq not in seen:
            return seq
    return seq


def _build_chain(rng: random.Random, hops: int, seen: set = None) -> Tuple[List[Tuple[str, str]], str]:
    """Construct one chain of `hops` edges. Returns (pairs, k0)."""
    hops = max(1, hops)
    nodes: List[str] = []
    used = set()
    while len(nodes) < hops + 1:
        candidate = _sample_seq(rng, seen)
        if candidate in used:
            continue
        if seen is not None and candidate in seen:
            continue
        nodes.append(candidate)
        used.add(candidate)
        if seen is not None:
            seen.add(candidate)
    pairs: List[Tuple[str, str]] = []
    for i in range(hops):
        pairs.append((nodes[i], nodes[i + 1]))
    return pairs, nodes[0]


def _build_qa_first(
    rng: random.Random,
    qa_budget: int,
    hops: int,
    enc,
    seen: set = None,
) -> Tuple[int, List[str], List[Tuple[str, str]]]:
    """Build QA section first; return (qa_len_tokens, qa_parts, chain_pairs)."""
    qa_header = "QA\n"
    qa_len = len(enc.encode_ordinary(qa_header))
    qa_parts: List[str] = []
    chain_pairs: List[Tuple[str, str]] = []

    attempts = 0
    while attempts < 5000:
        attempts += 1
        pairs, k0 = _build_chain(rng, hops, seen)
        v_goal = pairs[-1][1]
        item = f"?{k0}={v_goal}\n"
        need = len(enc.encode_ordinary(item))
        if qa_len + need > qa_budget:
            break
        qa_parts.append(item)
        qa_len += need
        chain_pairs.extend(pairs)
    return qa_len, qa_parts, chain_pairs


def _add_distractors(
    rng: random.Random,
    hops: int,
    target_pairs: List[Tuple[str, str]],
    max_pairs: int,
    seen: set = None,
) -> None:
    """Fill with extra chains or random pairs until max_pairs or attempts exhausted."""
    if seen is None:
        seen = set(target_pairs)
    else:
        seen.update(target_pairs)
    attempts = 0
    while len(target_pairs) < max_pairs and attempts < 4000:
        attempts += 1
        if rng.random() < 0.6:
            pairs, _ = _build_chain(rng, hops, seen)
            candidates = pairs
        else:
            k_seq = _sample_seq(rng, seen)
            v_seq = _sample_seq(rng, seen)
            candidates = [(k_seq, v_seq)]
        for pair in candidates:
            if pair in seen:
                continue
            seen.add(pair)
            seen.add(pair[0])
            seen.add(pair[1])
            target_pairs.append(pair)
            if len(target_pairs) >= max_pairs:
                break


def _render_pairs_to_db(
    rng: random.Random,
    pairs: List[Tuple[str, str]],
    db_budget: int,
    enc,
) -> Tuple[int, List[str]]:
    """Render shuffled (K,V) pairs into DB text with separators."""
    rng.shuffle(pairs)
    db_header = "DB\n"
    db_len = len(enc.encode_ordinary(db_header))
    db_parts: List[str] = []
    for k_seq, v_seq in pairs:
        frag = f"|{k_seq}->{v_seq}|"
        need = len(enc.encode_ordinary(frag))
        if db_len + need > db_budget:
            break
        db_parts.append(frag)
        db_len += need
    return db_len, db_parts


def build_episode(
    rng: random.Random,
    hops: int,
    max_hops: int,
    target_tokens: int,
    enc,
) -> Tuple[List[int], str]:
    """Build one episode (length target_tokens, without final EOT)."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    max_hops = max(hops, max_hops)
    db_budget = int(target_tokens * (2 * max_hops) / (2 * max_hops + 1))
    qa_budget = target_tokens - db_budget

    seen_seqs: set = set()
    qa_len, qa_parts, chain_pairs = _build_qa_first(rng, qa_budget, hops, enc, seen_seqs)

    approx_pair_tokens = MAX_SEQ_LEN * 2 + 4  # rough estimate for budgeting
    desired_pairs = max(len(chain_pairs), max(4, int(db_budget / approx_pair_tokens) + 2))
    _add_distractors(rng, hops, chain_pairs, desired_pairs, seen_seqs)

    db_len, db_parts = _render_pairs_to_db(rng, chain_pairs, db_budget, enc)

    episode_text = "DB\n" + ''.join(db_parts) + "\n" + "QA\n" + ''.join(qa_parts)
    ids = enc.encode_ordinary(episode_text)

    if len(ids) < target_tokens:
        filler = enc.encode_ordinary("#")
        if not filler:
            filler = [0]
        need = target_tokens - len(ids)
        reps = need // len(filler)
        rem = need % len(filler)
        ids.extend(filler * reps)
        ids.extend(filler[:rem])
    else:
        ids = ids[:target_tokens]

    _check_token_range(ids)
    return ids, episode_text


def _build_episode_worker(args_tuple) -> Tuple[List[int], str]:
    seed, ep_index, hops, max_hops, target_tokens = args_tuple
    rng = random.Random(seed + ep_index)
    if _ENC is None:
        _worker_init()
    return build_episode(rng, hops, max_hops, target_tokens, _ENC)


def write_split(
    out_path: str,
    episodes: int,
    seed: int,
    block_size: int,
    write_txt: bool,
    workers: int,
    hops: int,
    max_hops: int,
) -> None:
    """Write one split to .bin (uint16) and optional .txt."""
    if block_size <= 1:
        raise ValueError("BLOCK_SIZE must be > 1 (needs room for EOT)")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    text_path = out_path.replace('.bin', '.txt')

    if workers is None or workers <= 1:
        _worker_init()
        rng = random.Random(seed)
        target_tokens = block_size - 1
        episode_iter: Iterable[Tuple[List[int], str]] = (
            build_episode(rng, hops, max_hops, target_tokens, _ENC)
            for _ in range(episodes)
        )
    else:
        _worker_init()
        target_tokens = block_size - 1
        args = [
            (seed, ep_idx, hops, max_hops, target_tokens)
            for ep_idx in range(episodes)
        ]
        chunksize = max(1, min(256, (episodes // (max(1, workers) * 8)) or 1))
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=workers, initializer=_worker_init)
        episode_iter = pool.imap(_build_episode_worker, args, chunksize=chunksize)

    with open(out_path, 'wb') as f_bin, (open(text_path, 'w', encoding='utf-8') if write_txt else nullcontext()) as f_txt:
        pbar = tqdm(total=episodes, desc=os.path.basename(out_path), leave=False)
        for episode_tokens, episode_text in episode_iter:
            episode_tokens.append(_ENC.eot_token)
            if len(episode_tokens) != block_size:
                raise AssertionError("episode length mismatch after appending EOT")
            _check_token_range(episode_tokens)
            arr = np.asarray(episode_tokens, dtype=np.uint16)
            arr.tofile(f_bin)
            if write_txt and f_txt is not None:
                f_txt.write(episode_text)
                f_txt.write("\n<|EOT|>\n\n")
            pbar.update(1)
        pbar.close()

    if workers is not None and workers > 1:
        pool.close()
        pool.join()


def run_smoke_tests(out_dir: str) -> None:
    """Lightweight correctness checks (episodes=2)."""
    print("Running smoke tests...")
    env_backup = os.environ.copy()
    os.environ.update({
        'TRAIN_EPISODES': '2',
        'VAL_EPISODES': '2',
        'SEED': '42',
        'BLOCK_SIZE': '64',
        'WRITE_TXT': '1',
        'WORKERS': '0',
        'OUT_DIR': out_dir,
        'HOPS': '2',
        'MAXHOPS': '2',
        'KV_VOCAB': '20',
        'SEP_VOCAB': '5',
    })
    try:
        main()
        train_bin = os.path.join(out_dir, 'train.bin')
        val_bin = os.path.join(out_dir, 'val.bin')
        for path, episodes in [(train_bin, 2), (val_bin, 2)]:
            arr = np.fromfile(path, dtype=np.uint16)
            expected = episodes * int(os.environ['BLOCK_SIZE'])
            assert arr.size == expected, f"size mismatch for {path}"
            for i in range(episodes):
                episode = arr[i * int(os.environ['BLOCK_SIZE']):(i + 1) * int(os.environ['BLOCK_SIZE'])].tolist()
                assert episode[-1] == _ENC.eot_token, "EOT missing"
                assert all(0 <= t <= UINT16_MAX_SAFE for t in episode)
        print("SMOKE TESTS PASS")
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def main() -> None:
    train_episodes = _env_int('TRAIN_EPISODES', 50_000)
    val_episodes = _env_int('VAL_EPISODES', 10_000)
    seed = _env_int('SEED', 1024)
    block_size = _env_int('BLOCK_SIZE', 1024)
    write_txt = _env_int('WRITE_TXT', 1) > 0
    workers_env = _env_int('WORKERS', 8)
    max_hops = _env_int('MAXHOPS', _env_int('HOPS', 2))
    hops = _env_int('HOPS', 2)
    kv_vocab = _env_int('KV_VOCAB', DEFAULT_KV_VOCAB)
    sep_vocab = _env_int('SEP_VOCAB', DEFAULT_SEP_VOCAB)

    try:
        cpu_cnt = mp.cpu_count()
    except Exception:
        cpu_cnt = workers_env
    workers = max(1, workers_env)
    if isinstance(cpu_cnt, int) and cpu_cnt > 0:
        workers = min(workers, cpu_cnt)

    out_dir = os.environ.get('OUT_DIR', os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')

    print("Generating kv_retrieval_multihop dataset (text-first)...")
    print(f"train episodes={train_episodes}, val episodes={val_episodes}, block_size={block_size}, hops={hops}, max_hops={max_hops}, seed={seed}, workers={workers}, write_txt={write_txt}")

    write_split(train_path, train_episodes, seed, block_size, write_txt, workers, hops, max_hops)
    write_split(val_path, val_episodes, seed + 1, block_size, write_txt, workers, hops, max_hops)

    # Encoder for meta
    enc = tiktoken.get_encoding('gpt2')
    pad_token = enc.encode_ordinary("#")[0]
    vocab_size = enc.n_vocab
    def _compute_mh_bounds(bin_path: str, block_size: int):
        arr = np.fromfile(bin_path, dtype=np.uint16)
        if arr.size == 0:
            return {'qa_start': [], 'pad_start': []}
        num_eps = arr.size // block_size
        data = arr[:num_eps * block_size].reshape(num_eps, block_size)
        qa_header = enc.encode_ordinary("\nQA\n")
        pat_len = len(qa_header)
        qa_start = np.full(num_eps, block_size // 2, dtype=np.int64)
        pad_start = np.full(num_eps, block_size - 1, dtype=np.int64)
        from tqdm.auto import tqdm as _tqdm
        desc = f"mh_bounds {os.path.basename(bin_path)}"
        for ep in _tqdm(range(num_eps), desc=desc, leave=False):
            row = data[ep]
            idx_pad = np.where(row == pad_token)[0]
            if idx_pad.size > 0:
                pad_start[ep] = int(idx_pad[0])
            if pat_len > 0 and pat_len < block_size:
                for pos in range(block_size - pat_len):
                    if np.array_equal(row[pos:pos + pat_len], qa_header):
                        qa_start[ep] = pos + pat_len
                        break
        return {'qa_start': qa_start.tolist(), 'pad_start': pad_start.tolist()}

    # compute mh bounds for train/val and include in meta so train.py can load them directly
    train_bounds = _compute_mh_bounds(train_path, block_size)
    val_bounds = _compute_mh_bounds(val_path, block_size)

    meta = {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'desc': 'kv_retrieval_multihop synthetic dataset (text-first). DB: shuffled K->V pairs; QA: ?K0=V_goal with multihop chains.',
        'format_version': 2,
        'special_tokens': {
            'PAD': pad_token,
            'EOT': enc.eot_token,
            'ANSWER_DELIM': enc.encode_ordinary('=')[0] if enc.encode_ordinary('=') else None,
            'NEWLINE': enc.encode_ordinary('\n')[0] if enc.encode_ordinary('\n') else None,
        },
        'mh_bounds': {
            'train': train_bounds,
            'val': val_bounds,
        },
        'hops': hops,
        'max_hops': max_hops,
        'kv_vocab': kv_vocab,
        'sep_vocab': sep_vocab,
    }
    meta_path = os.path.join(out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f_meta:
        pickle.dump(meta, f_meta)
    print(f"Wrote meta to {meta_path} (vocab_size={vocab_size})")


if __name__ == '__main__':
    if '--test' in sys.argv or os.environ.get('RUN_SMOKE_TEST'):
        run_smoke_tests(os.environ.get('OUT_DIR', os.path.join(os.path.dirname(__file__), 'test_out')))
    else:
        main()
