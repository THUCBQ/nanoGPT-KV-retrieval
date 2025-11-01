# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'k-judge'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'k_judge'
tokens_per_pass = 32768
gradient_accumulation_steps = 1
block_size = 1024  # context of up to 1024 previous characters
batch_size = tokens_per_pass // block_size

# baby GPT model :)
n_layer = 2
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 2e-4 # with baby networks can afford to go a bit higher
max_iters = 320000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

wandb_log = True # override via command line if you like
wandb_project = 'nanoGPT-kjudge'
wandb_run_name = f'KV-L{n_layer}-SEQLEN{block_size}'