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

from model import GPTConfig, GPT

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
if dataset in ('k_judge', ):
    # keep different block_size datasets under separate subdirectories
    data_dir = os.path.join(data_dir, f"bs{block_size}")
    os.makedirs(data_dir, exist_ok=True)

# Auto-generation for k_judge (mirrors kv_retrieval behavior, separate bs subdirs)
def _ensure_k_judge_data(data_dir: str, block_size: int):
    if dataset not in ('k_judge', ):
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
            env['WRITE_TXT'] = '1'
            env['OUT_DIR'] = os.path.abspath(data_dir)
            cmd = [sys.executable, prep_path]
            subprocess.run(cmd, env=env, check=True)
            print(f"[{dataset}] Generation complete.")
        else:
            for _ in range(600):
                if all(os.path.exists(p) for p in (train_bin, val_bin)):
                    break
                time.sleep(0.5)

_ensure_k_judge_data(data_dir, block_size)

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

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    if dataset in ('k_judge', ):
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
            j_sel = start_t + offset
            ix = (j_sel - block_size).to(torch.int64)
        else:
            # Fallback to uniform sampling once, with a one-time notice
            global _answer_pos_warned
            if split not in _answer_pos_warned:
                print(f"[{dataset}] Warning: no answer positions found for split '{split}', falling back to uniform sampling.")
                _answer_pos_warned.add(split)
            ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        # Uniform sampling over entire stream for all other datasets
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

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
    """Estimate loss and last-token accuracy over many batches."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accs_last = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            pred_tokens = torch.argmax(logits, dim=-1)
            acc_last = (pred_tokens[:, -1] == Y[:, -1]).float().mean().item()
            accs_last[k] = acc_last
        
        out[split] = {
            'loss': losses.mean().item(),
            'last_token_acc': accs_last.mean().item(),
        }
    model.train()
    return out

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
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
start_t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
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
        if iter_num > 0:
            elapsed_time = time.time() - start_t0
            total_estimated_time = elapsed_time / iter_num * max_iters
            print(f"step {iter_num}: estimated total training time: {total_estimated_time/3600:.2f} hours")
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            f"\ntrain last_acc {train_acc:.4f}, val last_acc {val_acc:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/last_token_acc": train_acc,
                "val/last_token_acc": val_acc,
                "best_val_loss": min(best_val_loss, val_loss) if best_val_loss is not None else val_loss,
                "total_tokens": iter_num * tokens_per_iter,
                "total_samples": iter_num * tokens_per_iter // block_size,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
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
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
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