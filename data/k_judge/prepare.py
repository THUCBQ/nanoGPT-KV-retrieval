# k_judge_synthetic_minimal_neg.py
# Minimal changes from original: simplified negative sampling via 1-char flip.
import os
import random
import string
import pickle
from typing import List, Tuple
from contextlib import nullcontext

import numpy as np
import tiktoken
import multiprocessing as mp
from tqdm.auto import tqdm

ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits  # compact tokenization-friendly
_ENC = None  # global encoder for worker processes

def _worker_init():
    global _ENC
    _ENC = tiktoken.get_encoding('gpt2')

def _worker_job(args):
    i, epi_seed, target_tokens = args
    rng = random.Random(epi_seed)
    ids = build_episode(rng=rng, enc=_ENC, target_tokens=target_tokens)
    return ids

def gen_random_str(rng: random.Random, length: int) -> str:
    return ''.join(rng.choices(ALPHABET, k=length))

def fit_db_string(
    rng: random.Random,
    enc,
    max_tokens: int,
) -> Tuple[str, List[int]]:
    """
    Create a random DB string whose tokenized length (including trailing \n) fits <= max_tokens.
    Returns the DB string (without trailing \n) and its token ids including a trailing \n.
    """
    # If no room, return empty
    if max_tokens <= 0:
        return "", []

    # Grow the DB string in character chunks, checking token budget each time
    s = ""
    ids: List[int] = []
    # heuristic chunk sizes: start bigger then taper if needed
    chunk_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    remaining_tokens = max_tokens

    # We'll repeatedly try to add chunks as long as they fit with a final \n
    def enc_with_newline(text: str) -> List[int]:
        return enc.encode_ordinary(text + "\n")

    # Try to add chunks while respecting token budget
    for cs in chunk_sizes:
        while True:
            candidate = s + gen_random_str(rng, cs)
            cand_ids = enc_with_newline(candidate)
            if len(cand_ids) <= max_tokens:
                s = candidate
                ids = cand_ids
            else:
                break

    # Ensure newline present in ids if any content; if empty and even "\n" doesn't fit, return empty
    if not s:
        # try to fit just a newline to separate visually
        nl_ids = enc.encode_ordinary("\n")
        if len(nl_ids) <= max_tokens:
            return "", nl_ids
        else:
            return "", []

    return s, ids

def build_episode(
    rng: random.Random,
    enc,
    target_tokens: int,
) -> List[int]:
    """
    Build one episode token ids (without final EOT), length exactly target_tokens.
    Composition: 50% DB (header + DB string + newline), 50% QA (header + multiple ?Q=A lines).
    """
    ids: List[int] = []

    # Split budget ~1:2 (DB:QA)
    db_target = target_tokens // 3
    qa_target = target_tokens - db_target

    # DB header (compact)
    db_header_ids = enc.encode_ordinary("DB\n")
    if len(db_header_ids) <= db_target:
        ids.extend(db_header_ids)
    # Remaining budget for DB body
    db_body_budget = max(0, db_target - len(ids))

    # Build DB body string to fit budget (including trailing \n)
    db_string, db_body_ids = fit_db_string(rng, enc, db_body_budget)
    if db_body_ids:
        ids.extend(db_body_ids)

    # Remaining tokens for QA including header
    remaining_for_qa_total = max(0, target_tokens - len(ids))

    # QA header (compact)
    qa_header_ids = enc.encode_ordinary("QA\n")
    if len(qa_header_ids) <= remaining_for_qa_total:
        ids.extend(qa_header_ids)
        remaining_for_qa_body = remaining_for_qa_total - len(qa_header_ids)
    else:
        remaining_for_qa_body = 0

    NEG_MAX_ATTEMPTS = 100
    MIN_Q = 8
    MAX_Q = 12

    def sample_positive_query() -> str:
        if not db_string:
            return ""  # signal no positive possible
        # sample a random substring of DB with length in [8, 12]
        min_q, max_q = MIN_Q, MAX_Q
        if len(db_string) < min_q:
            return ""  # cannot form positive of required length
        L = rng.randint(min_q, min(max_q, len(db_string)))
        start = rng.randint(0, len(db_string) - L)
        return db_string[start:start + L]

    def sample_negative_query() -> str:
        if not db_string:
            return gen_random_str(rng, rng.randint(MIN_Q, MAX_Q))

        for _ in range(NEG_MAX_ATTEMPTS):
            q = sample_positive_query()  # Get the string
                
            pos = rng.randrange(len(q))
            choices = ALPHABET.replace(q[pos], '')
            new_ch = rng.choice(choices)
            
            # Convert to list, modify, and convert back to string
            q_list = list(q)
            q_list[pos] = new_ch
            q_modified = ''.join(q_list)
            
            if db_string.find(q_modified) == -1:
                return q_modified

        return gen_random_str(rng, MIN_Q)

    # Target ~1:1 balance of answers; pick the class with fewer so far
    qa_count = 0
    pos_count = 0
    neg_count = 0
    while remaining_for_qa_body > 0:
        # choose which label to attempt next based on counts
        want_pos = pos_count <= neg_count
        if want_pos and db_string:
            q = sample_positive_query()
            a = '1'
            if not q:
                # cannot make positive, switch to negative
                q = sample_negative_query()
                a = '0'
        else:
            q = sample_negative_query()
            a = '0'

        line = f"?{q}={a}\n"
        line_ids = enc.encode_ordinary(line)
        if len(line_ids) <= remaining_for_qa_body:
            ids.extend(line_ids)
            remaining_for_qa_body -= len(line_ids)
            qa_count += 1
            if a == '1':
                pos_count += 1
            else:
                neg_count += 1
        else:
            break

    # Pad to exact target with benign filler
    if len(ids) < target_tokens:
        filler = "#"
        pad_ids = enc.encode_ordinary(filler)
        remaining = target_tokens - len(ids)
        reps = remaining // len(pad_ids)
        rem = remaining % len(pad_ids)
        ids.extend(pad_ids * reps)
        if rem:
            ids.extend(pad_ids[:rem])

    assert len(ids) == target_tokens, f"episode length {len(ids)} != target {target_tokens}"
    return ids

def write_split(
    out_path: str,
    enc,
    episodes: int,
    seed: int,
    block_size: int,
    write_txt: bool,
    workers: int,
):
    text_path = out_path.replace('.bin', '.txt')
    with open(out_path, 'wb') as f, (open(text_path, 'w', encoding='utf-8') if write_txt else nullcontext()) as ftxt:
        total_tokens = 0

        # worker setup: we create per-episode seeds for determinism
        target_tokens = block_size - 1

        if workers is None or workers <= 1:
            # sequential with tqdm
            rng = random.Random(seed)
            pbar = tqdm(total=episodes, desc=os.path.basename(out_path), leave=False)
            for i in range(episodes):
                ids = build_episode(rng=rng, enc=enc, target_tokens=target_tokens)
                ids_eot = ids + [enc.eot_token]
                assert len(ids_eot) == block_size
                arr = np.array(ids_eot, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(arr)
                if write_txt and ftxt is not None:
                    try:
                        text = enc.decode(ids)
                    except Exception:
                        text = ''.join(str(t) for t in ids)
                    ftxt.write(text)
                    ftxt.write("\n<|EOT|>\n\n")
                pbar.update(1)
            pbar.close()
        else:
            # parallel workers using ordered imap to keep episodes in order
            seeds = [(i, seed + i, target_tokens) for i in range(episodes)]
            # adaptive chunksize to reduce IPC overhead while keeping memory bounded
            adaptive_chunksize = max(1, min(256, (episodes // (workers * 8)) or 1))
            with mp.get_context('spawn').Pool(processes=workers, initializer=_worker_init) as pool:
                for ids in tqdm(pool.imap(_worker_job, seeds, chunksize=adaptive_chunksize), total=episodes, desc=os.path.basename(out_path), leave=False):
                    ids_eot = ids + [enc.eot_token]
                    assert len(ids_eot) == block_size
                    arr = np.array(ids_eot, dtype=np.uint16)
                    arr.tofile(f)
                    total_tokens += len(arr)
                    if write_txt and ftxt is not None:
                        try:
                            text = enc.decode(ids)
                        except Exception:
                            text = ''.join(str(t) for t in ids)
                        ftxt.write(text)
                        ftxt.write("\n<|EOT|>\n\n")

def main():
    TRAIN_EPISODES = int(os.environ.get('TRAIN_EPISODES', 10_000))
    VAL_EPISODES = int(os.environ.get('VAL_EPISODES', 10_000))
    SEED = int(os.environ.get('SEED', 1337))
    BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', 1024))
    WRITE_TXT = int(os.environ.get('WRITE_TXT', 1)) > 0
    # default workers=16 but cap by CPU count to avoid oversubscription
    default_workers = 16
    try:
        cpu_cnt = mp.cpu_count()
    except Exception:
        cpu_cnt = default_workers
    WORKERS = int(os.environ.get('WORKERS', default_workers))
    if WORKERS < 1:
        WORKERS = 1
    WORKERS = cpu_cnt if isinstance(cpu_cnt, int) and cpu_cnt > 0 else WORKERS

    enc = tiktoken.get_encoding('gpt2')

    dstdir = os.environ.get('OUT_DIR', os.path.dirname(__file__))
    os.makedirs(dstdir, exist_ok=True)
    train_path = os.path.join(dstdir, 'train.bin')
    val_path = os.path.join(dstdir, 'val.bin')

    print("Generating k_judge dataset... (minimal negative sampling: 1-char flip)")
    print(f"train episodes: {TRAIN_EPISODES}, val episodes: {VAL_EPISODES}")
    print(f"episode composition: ~1:2 DB:QA")
    print(f"block_size: {BLOCK_SIZE}, seed: {SEED}, write_txt: {WRITE_TXT}, workers: {WORKERS}")

    write_split(train_path, enc, TRAIN_EPISODES, SEED, BLOCK_SIZE, WRITE_TXT, WORKERS)
    write_split(val_path, enc, VAL_EPISODES, SEED + 1, BLOCK_SIZE, WRITE_TXT, WORKERS)

    meta = {
        'vocab_size': enc.n_vocab,
        'block_size': BLOCK_SIZE,
        'desc': 'k_judge synthetic dataset (DB: long random string, QA: ?Q=A with substring membership, Q len 8-12, minimal neg sampling via 1-char flip)',
        'format_version': 2,
    }
    with open(os.path.join(dstdir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    main()
