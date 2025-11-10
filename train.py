"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import sys
import subprocess
import tiktoken
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Dict, List

from model import GPTConfig, GPT

from torch.nn import functional as F

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768 
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# whether to compute loss only on answer spans for kv_retrieval / k_judge datasets
use_masked_answer_loss = True
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
# torch.backends.cuda.matmul.fp32_precision = 'tf32'     # allow tf32 on matmul
# torch.backends.cudnn.conv.fp32_precision = 'tf32'      # allow tf32 on cudnn
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
if dataset in ('k_judge', 'kv_retrieval'):
    # keep different block_size datasets under separate subdirectories
    data_dir = os.path.join(data_dir, f"bs{block_size}")
    os.makedirs(data_dir, exist_ok=True)
elif dataset == 'kv_retrieval_multihop':
    # use both block_size and HOPS to namespace the dataset folder
    try:
        hops_cfg = int(globals().get('HOPS', os.environ.get('HOPS', 4)))
    except Exception:
        hops_cfg = int(os.environ.get('HOPS', 4))
    data_dir = os.path.join(data_dir, f"bs{block_size}_h{hops_cfg}")
    os.makedirs(data_dir, exist_ok=True)

# Auto-generation for k_judge and kv_retrieval
def _ensure_retrieval_data(data_dir: str, block_size: int):
    if dataset not in ('k_judge', 'kv_retrieval', 'kv_retrieval_multihop'):
        return
    prep_path = os.path.join('data', dataset, 'prepare.py')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin = os.path.join(data_dir, 'val.bin')
    train_txt = train_bin.replace('.bin', '.txt')
    val_txt = val_bin.replace('.bin', '.txt')

    need_regen = False
    meta_ok = False
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                meta_chk = pickle.load(f)
            meta_block_size = meta_chk.get('block_size', None)
            meta_fmt = meta_chk.get('format_version', None)
            if dataset == 'kv_retrieval_multihop':
                try:
                    hops_cfg = int(globals().get('HOPS', os.environ.get('HOPS', 4)))
                except Exception:
                    hops_cfg = int(os.environ.get('HOPS', 4))
                meta_hops = meta_chk.get('hops', None)
                if meta_block_size == block_size and meta_fmt == 2 and meta_hops == hops_cfg:
                    meta_ok = True
                else:
                    if master_process:
                        print(f"[{dataset}] meta mismatch. meta: block_size={meta_block_size}, hops={meta_hops}, format_version={meta_fmt}; expected block_size={block_size}, hops={hops_cfg}, format_version=2. Will regenerate.")
            else:
                if meta_block_size == block_size and meta_fmt == 2:
                    meta_ok = True
                else:
                    if master_process:
                        print(f"[{dataset}] meta mismatch. meta: block_size={meta_block_size}, format_version={meta_fmt}; expected block_size={block_size}, format_version=2. Will regenerate.")
        except Exception:
            if master_process:
                print(f"[{dataset}] unable to read meta at {meta_path}, will regenerate dataset.")
    else:
        if master_process:
            print(f"[{dataset}] meta not found at {meta_path}, will check for existing binaries before regenerating.")

    bins_ok = os.path.exists(train_bin) and os.path.exists(val_bin)
    txts_ok = os.path.exists(train_txt) and os.path.exists(val_txt)
    if not bins_ok:
        need_regen = True
    else:
        need_regen = (not meta_ok)

    if need_regen:
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"{dataset} prepare script not found at {prep_path}")
        if master_process:
            print(f"[{dataset}] Generating dataset with BLOCK_SIZE={block_size} ...")
            env = os.environ.copy()
            env['BLOCK_SIZE'] = str(block_size)
            # For multihop, write txt for readability by default
            env['WRITE_TXT'] = '0'
            env['OUT_DIR'] = os.path.abspath(data_dir)
            # Pass HOPS if configured
            if 'HOPS' in globals():
                try:
                    env['HOPS'] = str(int(globals()['HOPS']))
                except Exception:
                    pass
            cmd = [sys.executable, prep_path]
            subprocess.run(cmd, env=env, check=True)
            print(f"[{dataset}] Generation complete.")
        else:
            for _ in range(600):
                if all(os.path.exists(p) for p in (train_bin, val_bin)):
                    break
                time.sleep(0.5)

_ensure_retrieval_data(data_dir, block_size)

# Answer-aware sampling for kv_retrieval
# We want the last token of Y (i.e., data[i+block_size]) to always lie
# inside the answer segment A of a QA line formed as "?k=v\n".
# We precompute all token indices that are within A (between '=' and next '\n')
# and then sample i = j - block_size so that Y[-1] == data[j] is in A.
_answer_seg_cache = {}
_answer_pos_warned = set()

def _compute_answer_segments(bin_path: str, min_j: int):
    """Return tuple (starts, ends, cum_lengths) for answer segments within '?k=v\n'.
    Each segment is [start, end) with tokens strictly between '=' and next '\n'.
    min_j ensures we only consider j >= min_j to allow i=j-block_size >= 0.
    """
    enc = tiktoken.get_encoding('gpt2')
    eq_ids = enc.encode_ordinary('=')
    nl_ids = enc.encode_ordinary('\n')
    if len(eq_ids) != 1 or len(nl_ids) != 1:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    eq_id = eq_ids[0]
    nl_id = nl_ids[0]

    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    eq_idx = np.nonzero(data == eq_id)[0]
    if eq_idx.size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    nl_idx = np.nonzero(data == nl_id)[0]
    if nl_idx.size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    pos_in_nl = np.searchsorted(nl_idx, eq_idx + 1, side='left')
    starts = []
    ends = []
    lengths = []
    for e, p in zip(eq_idx, pos_in_nl):
        if p >= nl_idx.size:
            break
        nl = int(nl_idx[p])
        s = int(e) + 1
        # enforce j >= min_j
        s = max(s, min_j)
        if s < nl:
            starts.append(s)
            ends.append(nl) # exclusive
            lengths.append(nl - s)
    if not starts:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    cum = np.cumsum(lengths)
    return (starts, ends, cum)

def _get_answer_segments(split: str):
    if split in _answer_seg_cache:
        return _answer_seg_cache[split]
    bin_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    seg = _compute_answer_segments(bin_path, min_j=block_size)
    _answer_seg_cache[split] = seg
    return seg

# For kv_retrieval_multihop: cache answer token absolute indices (positions of label token)
_mh_answer_pos_cache = {}

def _mh_get_answer_positions(split: str):
    if split in _mh_answer_pos_cache:
        return _mh_answer_pos_cache[split]
    bin_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    meta_path_local = os.path.join(data_dir, 'meta.pkl')
    try:
        with open(meta_path_local, 'rb') as f_meta:
            meta_local = pickle.load(f_meta)
        ans_start_token = int(meta_local['special_tokens']['ANS_START'])
    except Exception:
        return np.array([], dtype=np.int64)
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    # answer at j when data[j-1] == ANS_START; ignore j < block_size for valid sampling
    idx_ans_start = np.nonzero(data == ans_start_token)[0]
    if idx_ans_start.size == 0:
        arr = np.array([], dtype=np.int64)
    else:
        j_positions = idx_ans_start + 1
        # filter within bounds
        j_positions = j_positions[(j_positions >= block_size) & (j_positions < len(data))]
        arr = j_positions.astype(np.int64)
    _mh_answer_pos_cache[split] = arr
    return arr

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if dataset in ('k_judge', 'kv_retrieval'):
        # Prefer answer-aware sampling so that Y's last token is inside an answer 'A'
        starts, ends, cum = _get_answer_segments(split)
        if cum.size > 0:
            total = int(cum[-1])
            # sample positions in [0, total)
            r = torch.randint(total, (batch_size,), dtype=torch.int64)
            # map to segment via searchsorted over cum lengths
            # convert to numpy for searchsorted then back; batch size small so overhead negligible
            seg_idx = np.searchsorted(cum, r.cpu().numpy(), side='right')
            seg_idx = torch.from_numpy(seg_idx).to(torch.int64)
            prev_cum = torch.zeros_like(seg_idx)
            prev_cum[seg_idx > 0] = torch.from_numpy(cum[:-1]).to(torch.int64)[seg_idx[seg_idx > 0]-1]
            offset = r - prev_cum
            start_t = torch.from_numpy(starts).to(torch.int64)[seg_idx]
            end_t = torch.from_numpy(ends).to(torch.int64)[seg_idx]
            j_sel = start_t + offset
            ix = (j_sel - block_size).to(torch.int64)
        else:
            # Fallback to uniform sampling once, with a one-time notice
            global _answer_pos_warned
            if split not in _answer_pos_warned:
                print(f"[{dataset}] Warning: no answer positions found for split '{split}', falling back to uniform sampling.")
                _answer_pos_warned.add(split)
            ix = torch.randint(len(data) - block_size, (batch_size,))
    elif dataset == 'kv_retrieval_multihop':
        # Answer-aware sampling: ensure Y[-1] is an answer label (immediately after ANS_START)
        j_positions = _mh_get_answer_positions(split)
        if j_positions.size > 0:
            # sample j uniformly from positions, then i = j - block_size
            sel = torch.randint(j_positions.size, (batch_size,), dtype=torch.int64)
            j_sel = torch.from_numpy(j_positions).to(torch.int64)[sel]
            ix = (j_sel - block_size).to(torch.int64)
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        # Uniform sampling over entire stream for all other datasets
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # compute answer spans or answer mask relative to Y for datasets that have answer targets
    # answer_spans may be either:
    # - None: no masking
    # - tuple(y_start_cpu, y_end_cpu): original interface for contiguous spans (k_judge, kv_retrieval)
    # - torch.BoolTensor mask of shape (B, T) on CPU marking positions in Y to include in loss (kv_retrieval_multihop)
    answer_spans = None
    if dataset in ('k_judge', 'kv_retrieval'):
        try:
            # ix is i (start index of X in the absolute data stream)
            # Y corresponds to absolute indices [i+1, i+block_size]
            # compute y_start = start_abs - (i+1), y_end = end_abs - (i+1)
            # clamp to [0, block_size]
            start_abs = start_t
            end_abs = end_t
            i_abs = ix.to(torch.int64)
            y_start = start_abs - (i_abs + 1)
            y_end = end_abs - (i_abs + 1)
            y_start = torch.clamp(y_start, min=0)
            y_end = torch.clamp(y_end, min=0, max=block_size)
            answer_spans = (y_start.cpu(), y_end.cpu())
        except Exception:
            answer_spans = None
    elif dataset == 'kv_retrieval_multihop':
        # For multihop dataset the answers are single-token labels that follow ANS_START markers inside X.
        # Create a per-sample boolean mask (on CPU) marking Y positions to include in the loss.
        try:
            meta_path_local = os.path.join(data_dir, 'meta.pkl')
            with open(meta_path_local, 'rb') as f_meta:
                meta_local = pickle.load(f_meta)
            ans_start_token = int(meta_local['special_tokens']['ANS_START'])
            # x is currently a CPU tensor of shape (B, T); answers to predict are Y[p] where X[p] == ANS_START
            mask = (x == ans_start_token)
            # mask dtype is bool on CPU; this will be interpreted by the masked loss function
            answer_spans = mask.cpu()
        except Exception:
            answer_spans = None

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # keep answer_spans as tuple of tensors on CPU, or boolean mask on CPU, or None.
    # Callers should move them to device as needed inside the loss helper.
    return x, y, answer_spans

# ---------------- Multihop answer token extraction -----------------
# For kv_retrieval_multihop, answer tokens immediately follow ANS_START markers.
# We treat each (ANS_START, label) as an answer span of length 1 (the label token).
def _extract_multihop_answer_spans(x_batch: torch.Tensor, ans_start_token: int) -> tuple:
    """Return (y_start, y_end) CPU tensors marking single-token answer spans in Y.

    We locate ANS_START positions inside X (which is length block_size) but answers to predict are next tokens in Y.
    For position p in X where X[p] == ANS_START, the predicted answer token is Y[p].
    So span relative to Y is [p, p+1).
    """
    B, T = x_batch.shape
    starts = []
    ends = []
    for b in range(B):
        row = x_batch[b]
        pos = (row == ans_start_token).nonzero(as_tuple=False).view(-1)
        for p in pos:
            p_int = int(p.item())
            if p_int < T:  # answer token exists at Y[p]
                starts.append(p_int)
                ends.append(p_int + 1)
    if not starts:
        return None
    # For uniform interface we need per-batch single span; but there can be multiple answers.
    # We'll compress all answer tokens into one merged multi-segment by setting min start and max end if contiguous.
    # However masked loss logic expects one span per batch element; easier: return None (no masking) and rely on metrics.
    # Instead, for metrics we will handle multihop separately; return None here.
    return None

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# track best metrics for wandb logging (both train and val splits)
best_train_last_token_acc = 0.0
best_val_last_token_acc = 0.0
best_train_answer_token_acc = 0.0
best_val_answer_token_acc = 0.0
best_train_answer_exact_match = 0.0
best_val_answer_exact_match = 0.0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_metrics():
    """Estimate loss, last-token accuracy, answer-token accuracy and answer exact-match over many batches.

    For kv_retrieval_multihop: answer tokens are labels directly after ANS_START markers (single-token answers).
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accs_last = torch.zeros(eval_iters)
        # answer-segment metrics
        ans_token_correct = 0
        ans_token_total = 0
        ans_exact_matches = 0
        ans_samples_with_answer = 0
        for k in range(eval_iters):
            X, Y, spans = get_batch(split)
            # For multihop dataset, identify answer tokens inline
            multihop_answer_positions = None
            if dataset == 'kv_retrieval_multihop':
                # Load special tokens from meta
                meta_path_local = os.path.join(data_dir, 'meta.pkl')
                try:
                    with open(meta_path_local, 'rb') as f_meta:
                        meta_local = pickle.load(f_meta)
                    ans_start_token = meta_local['special_tokens']['ANS_START']
                    # positions where X == ANS_START; Y at same index holds answer label
                    multihop_answer_positions = (X == ans_start_token).nonzero(as_tuple=False)
                except Exception:
                    multihop_answer_positions = None
            if use_masked_answer_loss:
                with ctx:
                    logits, _ = model(X, Y)
                # compute loss only on answer spans when available
                loss_masked = _masked_cross_entropy_on_answer(logits, Y, spans)
                losses[k] = loss_masked.item()
            else:
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            pred_tokens = torch.argmax(logits, dim=-1)
            acc_last = (pred_tokens[:, -1] == Y[:, -1]).float().mean().item()
            accs_last[k] = acc_last
            if dataset in ('k_judge', 'kv_retrieval') and spans is not None:
                y_start, y_end = spans
                T = Y.size(1)
                half = T // 2
                for b in range(Y.size(0)):
                    s = int(y_start[b].item())
                    e = int(y_end[b].item())
                    # restrict to second half only
                    s2 = max(s, half)
                    if e <= s2:
                        continue
                    tgt = Y[b, s2:e]
                    pred = pred_tokens[b, s2:e]
                    correct = (pred == tgt).sum().item()
                    total = tgt.numel()
                    ans_token_correct += correct
                    ans_token_total += total
                    if correct == total:
                        ans_exact_matches += 1
                    ans_samples_with_answer += 1
            elif dataset == 'kv_retrieval_multihop' and multihop_answer_positions is not None and multihop_answer_positions.numel() > 0:
                # multihop_answer_positions: rows of [batch_index, position]
                # Group by batch index to compute exact match per sample
                by_batch: Dict[int, List[int]] = {}
                T = Y.size(1)
                half = T // 2
                for row in multihop_answer_positions:
                    b_idx = int(row[0].item())
                    pos = int(row[1].item())
                    # only consider answers in the second half
                    if pos >= half:
                        by_batch.setdefault(b_idx, []).append(pos)
                for b, positions in by_batch.items():
                    positions.sort()
                    if len(positions) == 0:
                        continue
                    all_correct = True
                    for pos in positions:
                        tgt = Y[b, pos]
                        pred = pred_tokens[b, pos]
                        ans_token_total += 1
                        if int(pred.item()) == int(tgt.item()):
                            ans_token_correct += 1
                        else:
                            all_correct = False
                    ans_samples_with_answer += 1
                    if all_correct:
                        ans_exact_matches += 1

        # finalize answer-segment metrics
        if ans_token_total > 0:
            ans_token_acc = ans_token_correct / ans_token_total
        else:
            ans_token_acc = 0.0
        if ans_samples_with_answer > 0:
            ans_exact_match_rate = ans_exact_matches / ans_samples_with_answer
        else:
            ans_exact_match_rate = 0.0

        out[split] = {
            'loss': losses.mean().item(),
            'last_token_acc': accs_last.mean().item(),
            'answer_token_acc': ans_token_acc,
            'answer_exact_match': ans_exact_match_rate,
        }
    model.train()
    return out


def _masked_cross_entropy_on_answer(logits, targets, spans):
    """Compute cross entropy loss only on the answer span tokens provided by spans.

    logits: (B, T, V), targets: (B, T), spans: (y_start_cpu, y_end_cpu) or None.
    If spans is None or no tokens selected, falls back to full-token loss.
    """
    # logits: B,T,V ; targets: B,T
    if spans is None:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    # If spans is a boolean mask tensor (B, T) on CPU or device, use it directly
    if isinstance(spans, torch.Tensor) and spans.dtype == torch.bool:
        mask = spans.to(targets.device)
        # ensure mask has same shape
        if mask.dim() != targets.dim() or mask.size(0) != targets.size(0) or mask.size(1) != targets.size(1):
            # shape mismatch, fall back to full loss
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        # assume spans is the (y_start_cpu, y_end_cpu) tuple interface
        try:
            y_start, y_end = spans
        except Exception:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # ensure spans tensors are non-empty
        if not isinstance(y_start, torch.Tensor) or y_start.numel() == 0:
            # fallback to full loss
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        y_start = y_start.to(targets.device)
        y_end = y_end.to(targets.device)

        B, T = targets.shape
        mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)
        for b in range(B):
            # safe indexing in case shapes mismatch
            if b >= y_start.size(0):
                continue
            s = int(y_start[b].item())
            e = int(y_end[b].item())
            # clamp
            s = max(0, min(T, s))
            e = max(0, min(T, e))
            if e > s:
                mask[b, s:e] = True

    # if no positions selected, fall back to full loss
    if mask.sum() == 0:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    targets_masked = targets.clone()
    targets_masked[~mask] = -1
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_masked.view(-1), ignore_index=-1)
    return loss

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, spans = get_batch('train') # fetch the very first batch (keep spans to compute masked loss)
t0 = time.time()
start_t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

timestamp = time.strftime('%Y%m%d_%H%M%S')
log_root = os.path.join('outs', wandb_run_name if 'wandb_run_name' in globals() else 'run') + timestamp
os.makedirs(log_root, exist_ok=True)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        metrics = estimate_metrics()
        train_loss = metrics['train']['loss']
        val_loss = metrics['val']['loss']
        train_acc = metrics['train']['last_token_acc']
        val_acc = metrics['val']['last_token_acc']
        train_ans_token_acc = metrics['train'].get('answer_token_acc', 0.0)
        val_ans_token_acc = metrics['val'].get('answer_token_acc', 0.0)
        train_ans_exact = metrics['train'].get('answer_exact_match', 0.0)
        val_ans_exact = metrics['val'].get('answer_exact_match', 0.0)
        # update best-seen metrics (train/val)
        if train_acc > best_train_last_token_acc:
            best_train_last_token_acc = train_acc
        if val_acc > best_val_last_token_acc:
            best_val_last_token_acc = val_acc
        if train_ans_token_acc > best_train_answer_token_acc:
            best_train_answer_token_acc = train_ans_token_acc
        if val_ans_token_acc > best_val_answer_token_acc:
            best_val_answer_token_acc = val_ans_token_acc
        if train_ans_exact > best_train_answer_exact_match:
            best_train_answer_exact_match = train_ans_exact
        if val_ans_exact > best_val_answer_exact_match:
            best_val_answer_exact_match = val_ans_exact
        if iter_num > 0:
            elapsed_time = time.time() - start_t0
            total_estimated_time = elapsed_time / iter_num * max_iters
            print(f"step {iter_num}: estimated total training time: {total_estimated_time/3600:.2f} hours")
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            f"\ntrain last_acc {train_acc:.4f}, val last_acc {val_acc:.4f}"
            f"\ntrain ans_token_acc {train_ans_token_acc:.4f}, val ans_token_acc {val_ans_token_acc:.4f}"
            f"\ntrain ans_exact {train_ans_exact:.4f}, val ans_exact {val_ans_exact:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/last_token_acc": train_acc,
                "val/last_token_acc": val_acc,
                "train/answer_token_acc": train_ans_token_acc,
                "val/answer_token_acc": val_ans_token_acc,
                "train/answer_exact_match": train_ans_exact,
                "val/answer_exact_match": val_ans_exact,
                "best_val_loss": min(best_val_loss, val_loss) if best_val_loss is not None else val_loss,
                # log the best seen metrics (train and val)
                "best/train/last_token_acc": best_train_last_token_acc,
                "best/val/last_token_acc": best_val_last_token_acc,
                "best/train/answer_token_acc": best_train_answer_token_acc,
                "best/val/answer_token_acc": best_val_answer_token_acc,
                "best/train/answer_exact_match": best_train_answer_exact_match,
                "best/val/answer_exact_match": best_val_answer_exact_match,
                "total_tokens": iter_num * tokens_per_iter,
                "total_samples": iter_num * tokens_per_iter // block_size,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        # ---------------- Eval Sample Logging ----------------
        if master_process:
            # create unique eval log directory per run
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            log_path = os.path.join(log_root, f'eval_{iter_num}_{timestamp}.log')
            try:
                with open(log_path, 'w') as flog:
                    # Get a fresh eval batch for logging predictions
                    Xs, Ys, _sp = get_batch('val')
                    with torch.no_grad():
                        logits, _ = model(Xs, Ys)
                        preds = torch.argmax(logits, dim=-1)
                    max_show = min(5, Xs.size(0))
                    # Only output the last N tokens for readability
                    max_tokens_log = 100
                    for b in range(max_show):
                        x_tokens = Xs[b].cpu()
                        y_tokens = Ys[b].cpu()
                        p_tokens = preds[b].cpu()
                        if x_tokens.numel() > max_tokens_log:
                            x_tokens = x_tokens[-max_tokens_log:]
                        if y_tokens.numel() > max_tokens_log:
                            y_tokens = y_tokens[-max_tokens_log:]
                        if p_tokens.numel() > max_tokens_log:
                            p_tokens = p_tokens[-max_tokens_log:]
                        flog.write('X: ' + ' '.join(str(int(t)) for t in x_tokens) + '\n')
                        flog.write('Y: ' + ' '.join(str(int(t)) for t in y_tokens) + '\n')
                        flog.write('P: ' + ' '.join(str(int(t)) for t in p_tokens) + '\n')
                        flog.write('\n')
                print(f"Logged eval samples to {log_path}")
            except Exception as e:
                print(f"Eval logging failed: {e}")
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    # also store best-seen metrics so resume can continue logging them
                    'best_train_last_token_acc': best_train_last_token_acc,
                    'best_val_last_token_acc': best_val_last_token_acc,
                    'best_train_answer_token_acc': best_train_answer_token_acc,
                    'best_val_answer_token_acc': best_val_answer_token_acc,
                    'best_train_answer_exact_match': best_train_answer_exact_match,
                    'best_val_answer_exact_match': best_val_answer_exact_match,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        if use_masked_answer_loss:
            with ctx:
                logits, _ = model(X, Y)
            # compute loss only on answer spans
            loss = _masked_cross_entropy_on_answer(logits, Y, spans)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        else:
            with ctx:
                _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, spans = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

import os
os._exit(0)