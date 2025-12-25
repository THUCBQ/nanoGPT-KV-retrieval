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
# max_hops is used by data prep for budgeting; keep it equal to HOPS unless experimenting
MAXHOPS = 4
# maintain similar throughput across different block sizes
# Keep context modest for quick iteration; can increase once stable
tokens_per_pass = 16384
gradient_accumulation_steps = 1
block_size = 1024
batch_size = tokens_per_pass // block_size

# small GPT model for quicker convergence
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.0
attention_type = 'linear'
linear_backend = 'fla_deltanet' # 'fla_multiscale'|'fla_deltanet'|'fla_gateddeltanet

learning_rate = 4e-4
max_iters = 400000
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
wandb_run_name = ('Ful' if attention_type == 'causal' else
                  'GDe' if linear_backend == 'fla_gateddeltanet' else
                  'Del' if linear_backend == 'fla_deltanet' else
                  'MSc' if linear_backend == 'fla_multiscale' else '') + \
f'MHOP-L{n_layer}-H{n_head}-D{n_embd}-SEQLEN{block_size}-HOPS{HOPS}-MXH{MAXHOPS}'

# For multihop token-format dataset, use full loss by default (masked spans will be variable and sparse)
use_masked_answer_loss = False