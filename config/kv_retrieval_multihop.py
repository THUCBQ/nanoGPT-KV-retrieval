# Multi-hop KV retrieval training config

out_dir = 'kv-retrieval-multihop'
eval_interval = 500  # frequent because we'll likely overfit quickly
eval_iters = 100
log_interval = 100

always_save_checkpoint = False

# dataset and data loader
dataset = 'kv_retrieval_multihop'
# hops (chain length): each chain has length HOPS, final value is the goal
HOPS = 1
# maintain similar throughput across different block sizes
# Keep context modest for quick iteration; can increase once stable
tokens_per_pass = 16384
gradient_accumulation_steps = 1
block_size = 1024
batch_size = tokens_per_pass // block_size

# small GPT model for quicker convergence
n_layer = 4
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 2e-4
max_iters = 200000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
beta2 = 0.95

warmup_iters = 2000

# dtype/compile similar to other configs
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# logging
wandb_log = True
wandb_project = 'nanoGPT-KVretrieval-multihop'
wandb_run_name = f'MHOP-L{n_layer}-H{n_head}-D{n_embd}-SEQLEN{block_size}-HOPS{HOPS}'

# For multihop token-format dataset, use full loss by default (masked spans will be variable and sparse)
use_masked_answer_loss = True