"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --B=32 --compile=False

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
from contextlib import nullcontext
from tqdm import tqdm
import fire
import pandas as pd

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from soph.model import GPTConfig, GPT
from soph.utils.config_generator import FixedSeededNumpyRNG, FixedNumpySeed
import inspect

def get_calling_fname():
    stack = inspect.stack()
    
    # The 0th index is the current frame, 1st index is the caller
    caller_frame = stack[-1]  # The last frame is the entry point (main script)
    caller_filename = caller_frame.filename  # Full path to the main script

    # Extract the filename without extension
    caller_basename = os.path.splitext(os.path.basename(caller_filename))[0]
    return caller_basename

def basic_get_batch_fn(data_seed, data_dir, meta, BS,world_size=1):
    data = np.memmap(os.path.join(data_dir, 'data.bin'), dtype=np.uint8, mode='r', shape=(meta['num_samples'], meta['seq_len']))
    with FixedNumpySeed(data_seed):
        data_perm = np.random.permutation(world_size*BS*(meta['num_samples']//(BS*world_size))).reshape(-1,world_size, BS)
    def get_batch(iteration, rank): # the interface
        return data[data_perm[iteration % data_perm.shape[0]//world_size,rank]]
    return get_batch # (bs, N)

# def default_data_generator(data_seed, data_dir, meta, BS):
#     data = np.memmap(os.path.join(data_dir, 'data.bin'), dtype=np.uint8, mode='r', shape=(meta['num_samples'], meta['seq_len']))
#     my_rng = FixedSeededNumpyRNG(data_seed)
#     while True:
#         with my_rng:
#             ix = np.random.randint(meta['num_samples'], size=(BS,))
#         yield data[ix]

def train(*,out_dir='out', eval_interval=2000, log_interval=100, eval_iters=200,
          eval_only=False, save_checkpoint=False,
          init_from='scratch', wandb_log=True, wandb_project='CA', tag='',group='',
          dataset='openwebtext', A=4, B=128, L=12, D=768, dropout=0.0, bias=False,
          qk_norm=True, d_head=64, block_exponent=1, qk_exponent=1,
          lr=0.03, T=60000, wd=0, beta1=0.9, beta2=0.95, grad_clip=1.0,
          warmup=500, schedule='const', backend='nccl', device='cuda',
          dtype='auto', compile=True, apply_patch=None,
          logged_kwargs=None):
    # For backwards compatibility
    kwargs = locals().copy()
    kwargs.pop('dataset')
    # Data setup
    data_dir = f'./data/{dataset}'
    with open(os.path.join(data_dir, 'data.meta'), 'rb') as f:
        meta = pickle.load(f)
    world_size = int(os.environ['WORLD_SIZE']) if (int(os.environ.get('RANK', -1)) != -1) else 1
    get_batch = basic_get_batch_fn(seed, data_dir, meta,B,world_size)
    return train_basic(meta=meta,get_batch=get_batch,**kwargs)

def default_transform(data):
    x = torch.from_numpy((data[:,:-1]).astype(np.int64))
    y = torch.from_numpy((data[:,1:]).astype(np.int64))
    return x,y

def train_basic(*,out_dir='out', eval_interval=2000, log_interval=100, eval_iters=200,
          eval_only=False, save_checkpoint=False,
          init_from='scratch', wandb_log=True, wandb_project='CA', tag='',group='',
          dataset='openwebtext', A=4, B=128, L=12, D=768, dropout=0.0, bias=False,
          qk_norm=True, d_head=64, block_exponent=1, qk_exponent=1,
          lr=0.03, T=60000, wd=0, beta1=0.9, beta2=0.95, grad_clip=1.0,
          warmup=500, schedule='const', backend='nccl', device='cuda',
          dtype='auto', compile=True, apply_patch=None,get_batch=None, meta=None,
          log_history=False, log_every=10, logsteps=300, 
          logged_kwargs=None, return_log_as_df=False,transform_batch=default_transform, **kwargs): 
          # apply_patch is a function that takes a model and applies a patch to it in place
    config = locals().copy()
    logged_kwargs = logged_kwargs or {}
    config.update(logged_kwargs)
    config.pop('logged_kwargs')
    group = config.pop('group') or get_calling_fname()
    #if log_history:
    log_iters = np.round(np.linspace(warmup,T-1,logsteps)).astype(int)
    # log_iters2 = np.round(1.3**np.linspace(warmup,math.log(T,1.3),logsteps//2))
    # log_iters2 = np.round(np.geomspace(np.max(1, warmup),T-1,logsteps//2)).astype(int)

    log_iters_all = log_iters #np.unique(np.concatenate([log_iters,log_iters2])).astype(int)
    history_iters = log_iters[::log_every] #np.unique(np.concatenate([log_iters[::8],log_iters2[::8]])).astype(int)
    loss_history_2d = np.full((len(history_iters), len(log_iters_all)), np.nan) # (t_old, t_new)

    log_df = {
        'iter': [],
        'compute': [],
        'tokens': [],
        'train_loss': [],
        'train_loss_after': [],
        'lr': [],
    }

    def is_ampere_or_newer():
        if not torch.cuda.is_available():
            return False
        gpu_name = torch.cuda.get_device_name().lower()
        return any(x in gpu_name for x in ['a100', 'h100'])

    # Set data type
    if dtype == 'auto':
        dtype = 'bfloat16' if (torch.cuda.is_available() and 
                      torch.cuda.is_bf16_supported() and
                      is_ampere_or_newer()) else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(f'cuda:{ddp_local_rank}')
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert A % ddp_world_size == 0
        A //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    seed = 1337 + seed_offset
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    V = meta['vocab_size'] # do something less hacky later, though this doesn't hurt things now (but slight change in # params).
    #print(f"vocab = {V}")
    S = meta['seq_len']
    tokens_per_iter = A * ddp_world_size * B * S
    config['tokens_per_iter'] = tokens_per_iter
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    cached_data = {}
    
    def full_get_batch(iteration):
        data = get_batch(iteration, rank=ddp_rank)
        objs = [o.pin_memory().to(device, non_blocking=(device_type=='cuda')) for o in transform_batch(data)]
        return objs

    # Model init
    model_args = dict(L=L, D=D, S=S, V=V, bias=bias, dropout=dropout, qk_norm=qk_norm, d_head=d_head, block_exponent=block_exponent, qk_exponent=qk_exponent)
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
    else:
        raise ValueError("Invalid init_from value")
    if apply_patch is not None:
        apply_patch(model)

    P = sum(p.numel() for p in model.parameters())
    config['P'] = P

    if wandb_log and master_process:
        import wandb
        resume = "never" if (init_from != 'resume') else "allow"
        wandb.init(project=wandb_project, config=config, tags=[tag,group] if tag else [group], resume=resume)

    model.to(device)
    torch.cuda.empty_cache()
    
    

    # GradScaler and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer, peak_lrs = model.configure_optimizers(wd, lr, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    # Compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Wrap model into DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Training loop
    X, Y = full_get_batch(0)
    t0 = time.time()
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    train_loss_sum = 0
    train_loss_after_sum = 0
    elapsed=0
    best_val_loss = float('inf')

    # for iter_num in tqdm(range(T), disable=not master_process):
    for iter_num in (pbar := tqdm(range(T), disable=not master_process)):
        # Determine and set the learning rate
        lr_schedule = get_lr_schedule(iter_num, warmup, schedule, T)
        for param_group in optimizer.param_groups:
            param_group['lr'] = peak_lrs[param_group['name']] * lr_schedule

        # Evaluate and checkpoint
        if iter_num % eval_interval == 0 and master_process:
            if save_checkpoint:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        logdict = {}
        if iter_num == 0 and eval_only:
            break

        if (iter_num in log_iters_all) and master_process:
            
            if log_history:
                #model.eval()
                t_new_idx = np.where(log_iters_all == iter_num)[0][0]
                # old_iters = [k for k in history_iters if k < iter_num]
                old_iters = history_iters
                for t_old_idx, iteration in enumerate(old_iters):
                    loss_on_old = 0
                    with torch.no_grad(), ctx:
                        for micro_step in range(A):
                            Xp, Yp = full_get_batch(int(micro_step+A*iteration))
                            loss_on_old += model(Xp, Yp)[1].item() / A
                    logdict[f"hist_batch_{iteration}"] = loss_on_old
                    # loss_history_2d(t_old, t_new) = loss on t_old AFTER updating the model on t_new
                    # loss_history_2d(t, t) = loss on batch t immediately after updating the model on batch t
                    loss_history_2d[t_old_idx, t_new_idx] = loss_on_old

        # Training step
        for micro_step in range(A):
            if ddp:
                model.require_backward_grad_sync = (micro_step == A - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / A
            Xold, Yold = X, Y
            X, Y = full_get_batch(micro_step+A*iter_num+1)
            scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), ctx:
            loss_after = model(Xold, Yold)[1].item()
        # Logging
        train_loss_sum += loss.item() * A
        train_loss_after_sum += loss_after
        elapsed += 1
        t1 = time.time()#
        dt = t1 - t0
        t0 = t1
        if (iter_num in log_iters_all) and master_process:
            # logdict = {}
            # if log_history:
            #     #model.eval()
            #     t_new_idx = np.where(log_iters_all == iter_num)[0][0]
            #     # old_iters = [k for k in history_iters if k < iter_num]
            #     old_iters = history_iters
            #     for t_old_idx, iteration in enumerate(old_iters):
            #         loss_on_old = 0
            #         with torch.no_grad(), ctx:
            #             for micro_step in range(A):
            #                 Xp, Yp = full_get_batch(int(micro_step+A*iteration))
            #                 loss_on_old += model(Xp, Yp)[1].item() / A
            #         logdict[f"hist_batch_{iteration}"] = loss_on_old
            #         # loss_history_2d(t_old, t_new) = loss on t_old AFTER updating the model on t_new
            #         # loss_history_2d(t, t) = loss on batch t immediately after updating the model on batch t
            #         loss_history_2d[t_old_idx, t_new_idx] = loss_on_old
                #model.train()
            current_logs = {
                "iter": iter_num,
                "compute": 6 * P * tokens_per_iter * iter_num,
                "tokens": iter_num * tokens_per_iter,
                "train_loss": train_loss_sum / elapsed, # running avg of loss on current batch
                "train_loss_after": train_loss_after_sum / elapsed, # running avg of loss on previous batch
                "lr": lr_schedule * lr,
                "mfu": running_mfu * 100,
                **logdict
            }
            if wandb_log:
                wandb.log(current_logs)
            for k, v in current_logs.items():
                if k in log_df:
                    log_df[k].append(v)
            pbar.set_description(f"L: {train_loss_sum / elapsed:.3f}")
            # if train_loss_sum / elapsed < 1e-3:
            #     break
            train_loss_sum = 0
            train_loss_after_sum = 0
            elapsed = 0
    if wandb_log and master_process:
        wandb.finish() # was missing before??
    if ddp:
        destroy_process_group()
    
    return pd.DataFrame(log_df), loss_history_2d, history_iters

def get_lr_schedule(t, warmup, schedule, T):
    if t < warmup:
        return t / warmup
    if schedule == 'const':
        return 1.0
    elif schedule == 'linear':
        return 1.0 - (t - warmup) / (T - warmup)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

# removed entirely
# def estimate_loss(eval_iters, get_batch, model, ctx):
#     out = {}
#     model.eval()
#     for split in ['train']: # what is val doing here exactly, removed for now
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             with ctx:
#                 _, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

if __name__ == "__main__":
    fire.Fire(train)
