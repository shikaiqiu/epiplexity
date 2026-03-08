# Training script supporting single-GPU and DDP multi-GPU training.

import os
import time
import math
from contextlib import nullcontext
from tqdm import tqdm
import fire
import pandas as pd
import copy

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

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

def _align_loss_mask(mask_np, data_np, target_shape):
    if mask_np is None:
        return None
    if mask_np.shape == target_shape:
        return mask_np
    if mask_np.shape == data_np.shape:
        return mask_np[:, 1:]
    raise ValueError("loss_mask must match either data or target shape")


def default_transform(batch):
    if isinstance(batch, dict):
        data_np = batch['data']
        loss_mask_np = batch.get('loss_mask')
        extras = {k: batch[k] for k in batch.keys() - {'data', 'loss_mask'}}
    else:
        data_np = batch
        loss_mask_np = None
        extras = {}

    data_tensor = torch.from_numpy(data_np.astype(np.int64))
    inputs = data_tensor[:, :-1]
    targets = data_tensor[:, 1:]

    result = {
        'inputs': inputs,
        'targets': targets,
    }

    if loss_mask_np is not None:
        mask_np = _align_loss_mask(loss_mask_np, data_np, targets.shape)
        result['loss_mask'] = torch.from_numpy(mask_np.astype(np.bool_))

    if isinstance(batch, dict):
        if extras:
            raw_payload = {'data': data_tensor}
            for key, value in extras.items():
                if isinstance(value, np.ndarray):
                    raw_payload[key] = torch.from_numpy(value)
            result['raw'] = raw_payload
        else:
            result['raw'] = data_tensor
    else:
        result['raw'] = data_tensor

    return result


def train(*,out_dir='out', eval_interval=2000, log_interval=100, eval_iters=200,
          eval_only=False, save_checkpoint=False,
          init_from='scratch', wandb_log=True, wandb_project='CA', tag='',group='',
          arch='gpt', dataset='openwebtext', A=4, B=128, B_tokens=None, L=12, D=768, dropout=0.0,
          bias=False, qk_norm=True, d_head=64, block_exponent=1, qk_exponent=1,
          lr=0.03, T=60000, wd=0, beta1=0.9, beta2=0.95, grad_clip=1.0,
          warmup=500, schedule='const', backend='nccl', device='cuda',
          dtype='auto', compile=True, apply_patch=None,get_batch=None, meta=None,
          log_geometric=False, logsteps=300,
          logged_kwargs=None, return_log_as_df=False,
          transform_batch=None, return_model=False, requential=False, student_speed=1.0,
          max_kl=None, ema_steps=0, eval_callback=None, **kwargs):
          # apply_patch is a function that takes a model and applies a patch to it in place
    config = locals().copy()
    logged_kwargs = logged_kwargs or {}
    config.update(logged_kwargs)
    config.pop('logged_kwargs')
    group = config.pop('group') or get_calling_fname()
    log_iters = np.round(np.linspace(0,T-1,logsteps)).astype(int)
    if log_geometric:
        geom_iters = np.round(np.geomspace(1,T-1,logsteps)).astype(int)
        log_iters = np.unique(np.concatenate([log_iters, geom_iters]))

    if transform_batch is None:
        transform_batch = default_transform

    log_df = {
        'iter': [],
        'compute': [],
        'tokens': [],
        'train_loss': [],
        'train_loss_after': [],
        'lr': [],
        'K_auc': [],
        'K_req': [],
        'student_tokens': [],
        'student_compute': [],
        'student_loss': [],
        'distill_kl': [],
        'teacher_acc': [],
        'student_acc': [],
        'ema_train_loss': [],
        'ema_student_loss': [],
        'ema_teacher_acc': [],
        'ema_student_acc': [],
    }

    def build_student_inputs(entry):
        if entry['extra_inputs']:
            inputs = {'tokens': entry['tokens']}
            inputs.update(entry['extra_inputs'])
            return inputs
        return entry['tokens']

    def gather_synthetic_batches(raw_batches, teacher_for_synth, rounds=1):
        per_round_batches = []
        for _ in range(max(1, rounds)):
            round_entries = []
            for raw_batch in raw_batches:
                if raw_batch is None:
                    continue
                with torch.no_grad():
                    synthetic = teacher_for_synth.generate_synthetic(raw_batch, temperature=1.0)
                synthetic_data = synthetic['data']
                teacher_logits = synthetic['logits']
                mask = synthetic['mask']
                extra_inputs = {k: v for k, v in synthetic.items() if k not in ('data', 'logits', 'mask', 'tokens', 'inputs')}
                round_entries.append({
                    'tokens': synthetic_data[:, :-1],
                    'targets': synthetic_data[:, 1:],
                    'teacher_logits': teacher_logits,
                    'mask': mask,
                    'extra_inputs': extra_inputs,
                })
            if round_entries:
                per_round_batches.append(round_entries)
        if not per_round_batches:
            raise RuntimeError("Requential training requires synthetic batches, but none were generated.")
        return per_round_batches

    def average_student_teacher_kl(student_module, batches):
        total_kl_sum = 0.0
        total_tokens = 0
        with torch.no_grad():
            for entry in batches:
                mask = entry['mask']
                if mask is None or not mask.any():
                    continue
                student_inputs = build_student_inputs(entry)
                with ctx:
                    student_logits, _ = student_module(student_inputs, entry['targets'])
                mask_flat = mask.view(-1)
                token_count = mask_flat.long().sum().item()
                if token_count == 0:
                    continue
                teacher_logits_selected = entry['teacher_logits'].view(-1, entry['teacher_logits'].size(-1))[mask_flat]
                student_logits_selected = student_logits.view(-1, student_logits.size(-1))[mask_flat]
                teacher_probs = torch.softmax(teacher_logits_selected, dim=-1)
                student_log_probs = torch.log_softmax(student_logits_selected, dim=-1)
                kl_values = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-9) - student_log_probs), dim=-1)
                total_kl_sum += kl_values.sum().item()
                total_tokens += token_count
        if total_tokens == 0:
            return 0.0
        return total_kl_sum / total_tokens

    def perform_student_updates(synthetic_batches, student_updates, teacher_iter):
        metrics = {
            'loss_total': 0.0,
            'loss_count': 0,
            'kl_total': 0.0,
            'kl_count': 0,
            'token_kl_sum': 0.0,
            'student_tokens': 0,
        }
        student.train()
        lr_step = min(teacher_iter, T - 1)
        student_lr_schedule = get_lr_schedule(lr_step, warmup, schedule, T)
        for round_entries in synthetic_batches[:student_updates]:
            for param_group in student_optimizer.param_groups:
                param_group['lr'] = student_peak_lrs[param_group['name']] * student_lr_schedule
            student_optimizer.zero_grad(set_to_none=True)
            for entry in round_entries:
                student_inputs = build_student_inputs(entry)
                syn_targets = entry['targets']
                mask = entry['mask']
                with ctx:
                    student_logits, student_loss = student(student_inputs, syn_targets)
                    student_loss = student_loss / A
                student_scaler.scale(student_loss).backward()
                metrics['loss_total'] += student_loss.item() * A
                metrics['loss_count'] += 1
                if mask is not None and mask.any():
                    mask_flat = mask.view(-1)
                    token_count = mask_flat.long().sum().item()
                    if token_count > 0:
                        teacher_logits_selected = entry['teacher_logits'].view(-1, entry['teacher_logits'].size(-1))[mask_flat]
                        student_logits_selected = student_logits.detach().view(-1, student_logits.size(-1))[mask_flat]
                        teacher_probs = torch.softmax(teacher_logits_selected, dim=-1)
                        student_log_probs = torch.log_softmax(student_logits_selected, dim=-1)
                        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-9) - student_log_probs), dim=-1).mean()
                        metrics['kl_total'] += kl.item()
                        metrics['kl_count'] += 1
                        metrics['token_kl_sum'] += kl.item() * token_count
                        metrics['student_tokens'] += token_count
            student_scaler.step(student_optimizer)
            student_scaler.update()
            student_optimizer.zero_grad(set_to_none=True)
            if ema_student is not None:
                student_module_for_ema = student.module if ddp else student
                update_ema_model(ema_student, student_module_for_ema, ema_decay)
        return metrics

    def compute_masked_accuracy(preds, targets, mask):
        if mask is None:
            return (preds == targets).float().mean().item()
        mask_bool = mask.to(torch.bool)
        total = mask_bool.sum().float()
        if total.item() == 0:
            return float('nan')
        correct = ((preds == targets) & mask_bool).float().sum()
        return (correct / total).item()

    def update_ema_model(ema_model, source_model, decay):
        if ema_model is None:
            return
        with torch.no_grad():
            for ema_param, src_param in zip(ema_model.parameters(), source_model.parameters()):
                ema_param.mul_(decay).add_(src_param, alpha=1.0 - decay)
            for ema_buffer, src_buffer in zip(ema_model.buffers(), source_model.buffers()):
                ema_buffer.copy_(src_buffer)

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
    non_blocking = device_type == 'cuda'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    V = meta['vocab_size']
    S = meta['seq_len']
    tokens_per_iter = A * ddp_world_size * B * S
    config['tokens_per_iter'] = tokens_per_iter
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    def full_get_batch(iteration):
        raw_batch = get_batch(iteration, rank=ddp_rank)
        processed = transform_batch(raw_batch)
        if not isinstance(processed, dict):
            raise ValueError("transform_batch must return a dict with 'inputs' and 'targets'")

        try:
            inputs_cpu = processed['inputs']
            targets_cpu = processed['targets']
        except KeyError as err:
            raise KeyError("transform_batch output must include 'inputs' and 'targets'") from err

        def move_to_device(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                obj = torch.from_numpy(obj)
            if torch.is_tensor(obj):
                tensor = obj
                if non_blocking and tensor.device.type == 'cpu':
                    tensor = tensor.pin_memory()
                return tensor.to(device, non_blocking=non_blocking)
            if isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                converted = [move_to_device(v) for v in obj]
                return tuple(converted) if isinstance(obj, tuple) else converted
            return obj

        inputs = move_to_device(inputs_cpu)
        targets = move_to_device(targets_cpu)
        loss_mask_tensor = move_to_device(processed.get('loss_mask'))

        raw_payload = processed.get('raw')
        raw_tensor = None
        if requential:
            if raw_payload is None and isinstance(raw_batch, dict) and 'data' in raw_batch:
                raw_payload = torch.from_numpy(raw_batch['data'].astype(np.int64))
            raw_tensor = move_to_device(raw_payload)

        return {
            'inputs': inputs,
            'targets': targets,
            'loss_mask': loss_mask_tensor,
            'raw': raw_tensor,
        }

    # Model init
    model_args = dict(L=L, D=D, S=S, V=V, bias=bias, dropout=dropout, qk_norm=qk_norm, d_head=d_head, block_exponent=block_exponent, qk_exponent=qk_exponent)
    model = GPT(GPTConfig(**model_args))
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
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
    if requential:
        teacher_module = model.module if ddp else model
        student_raw = copy.deepcopy(teacher_module)
        student_raw.to(device)
        student_optimizer, student_peak_lrs = student_raw.configure_optimizers(wd, lr, (beta1, beta2), device_type)
        student_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        student_param_count = student_raw.get_num_params()
        if ddp:
            student = DDP(student_raw, device_ids=[ddp_local_rank])
        else:
            student = student_raw
    else:
        student = None
        student_raw = None
        student_optimizer = None
        student_peak_lrs = None
        student_scaler = None
        student_param_count = 0

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
    x0 = current_batch = full_get_batch(0)
    t0 = time.time()
    raw_model = model.module if ddp else model
    ema_decay = None
    ema_teacher = None
    ema_student = None
    if ema_steps and ema_steps > 1:
        ema_decay = math.exp(-1.0 / ema_steps)
        ema_teacher = copy.deepcopy(raw_model)
        ema_teacher.to(device)
        ema_teacher.eval()
        for p in ema_teacher.parameters():
            p.requires_grad_(False)
        if requential and student is not None:
            student_module_for_ema = student.module if ddp else student
            ema_student = copy.deepcopy(student_module_for_ema)
            ema_student.to(device)
            ema_student.eval()
            for p in ema_student.parameters():
                p.requires_grad_(False)

    running_mfu = -1.0
    train_loss_sum = 0
    train_loss_after_sum = 0
    elapsed=0
    best_val_loss = float('inf')
    cumulative_k_req = 0.0
    cumulative_student_tokens = 0.0
    last_student_kl = 0.0

    teacher_iter = 0
    loop_iter = 0
    pbar = tqdm(total=T, disable=not master_process)
    log_iter_set = set(log_iters)
    prev_features = None
    while teacher_iter < T:
        current_teacher_step = teacher_iter
        # Determine and set the learning rate based on completed teacher updates
        lr_schedule = get_lr_schedule(current_teacher_step, warmup, schedule, T)
        for param_group in optimizer.param_groups:
            param_group['lr'] = peak_lrs[param_group['name']] * lr_schedule

        # Training step
        teacher_raw_batches = []
        last_batch = current_batch
        for micro_step in range(A):
            if ddp:
                model.require_backward_grad_sync = (micro_step == A - 1)
            inputs = current_batch['inputs']
            targets = current_batch['targets']
            loss_mask = current_batch.get('loss_mask')
            with ctx:
                logits, loss = model(inputs, targets, loss_mask=loss_mask)
                loss = loss / A
            next_batch = full_get_batch(micro_step + A * loop_iter + 1)
            if requential and current_batch['raw'] is not None:
                teacher_raw_batches.append(current_batch['raw'])
            last_batch = current_batch
            current_batch = next_batch
            scaler.scale(loss).backward()

        distill_kl_logged = float('nan')
        teacher_blocked = False
        synthetic_batches = None
        student_updates_scheduled = 0

        if requential:
            raw_model_state = raw_model.training
            raw_model.eval()
            if student_speed > 0:
                if student_speed >= 1:
                    student_updates_scheduled = max(1, round(student_speed))
                else:
                    period = max(1, round(1 / student_speed))
                    if (current_teacher_step + 1) % period == 0:
                        student_updates_scheduled = 1
        decision_kl_value = last_student_kl
        if max_kl is not None:
            distill_kl_logged = decision_kl_value
            teacher_blocked = decision_kl_value > max_kl
        else:
            raw_model_state = False

        if not teacher_blocked:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_after = float('nan')
        if not teacher_blocked:
            if ema_teacher is not None:
                update_ema_model(ema_teacher, raw_model, ema_decay)
            with torch.no_grad(), ctx:
                eval_loss_mask = last_batch.get('loss_mask')
                loss_after = model(
                    last_batch['inputs'],
                    last_batch['targets'],
                    loss_mask=eval_loss_mask,
                )[1].item()

        student_loss_logged = float('nan')
        teacher_acc_logged = float('nan')
        student_acc_logged = float('nan')
        ema_train_loss_logged = float('nan')
        ema_student_loss_logged = float('nan')
        ema_teacher_acc_logged = float('nan')
        ema_student_acc_logged = float('nan')
        last_inputs = last_batch['inputs']
        last_targets = last_batch['targets']
        last_loss_mask = last_batch['loss_mask']
        student_tokens_iter = 0
        if requential:
            if (student_updates_scheduled > 0 or teacher_blocked):
                teacher_for_synth = ema_teacher if ema_teacher is not None else raw_model
                updates_to_run = student_updates_scheduled if not teacher_blocked else max(student_updates_scheduled, 1)
                synthetic_batches = gather_synthetic_batches(teacher_raw_batches, teacher_for_synth, rounds=updates_to_run if updates_to_run > 0 else 1)
            else:
                updates_to_run = 0
            if updates_to_run > 0:
                metrics = perform_student_updates(
                    synthetic_batches, updates_to_run, current_teacher_step
                )
                if metrics['loss_count'] > 0:
                    student_loss_logged = metrics['loss_total'] / metrics['loss_count']
                if metrics['kl_count'] > 0:
                    distill_kl_logged = metrics['kl_total'] / metrics['kl_count']
                    last_student_kl = distill_kl_logged
                if metrics['token_kl_sum'] > 0.0:
                    cumulative_k_req += metrics['token_kl_sum']
                student_tokens_iter += metrics['student_tokens']
            elif max_kl is not None:
                distill_kl_logged = last_student_kl
            if raw_model_state:
                raw_model.train()

        cumulative_student_tokens += student_tokens_iter

        if not teacher_blocked:
            eval_loss_mask = last_loss_mask
            with torch.no_grad(), ctx:
                teacher_logits, _ = raw_model(
                    last_inputs,
                    last_targets,
                    loss_mask=eval_loss_mask,
                )
            teacher_preds = teacher_logits.argmax(dim=-1)
            teacher_acc_logged = compute_masked_accuracy(teacher_preds, last_targets, last_loss_mask)

            if ema_teacher is not None:
                with torch.no_grad(), ctx:
                    ema_teacher_logits, ema_teacher_loss = ema_teacher(
                        last_inputs,
                        last_targets,
                        loss_mask=eval_loss_mask,
                    )
                if ema_teacher_loss is not None:
                    ema_train_loss_logged = ema_teacher_loss.item()
                ema_teacher_preds = ema_teacher_logits.argmax(dim=-1)
                ema_teacher_acc_logged = compute_masked_accuracy(ema_teacher_preds, last_targets, last_loss_mask)

            if requential and student is not None:
                student_module = student.module if ddp else student
                with torch.no_grad(), ctx:
                    student_logits, _ = student_module(
                        last_inputs,
                        last_targets,
                        loss_mask=eval_loss_mask,
                    )
                student_preds = student_logits.argmax(dim=-1)
                student_acc_logged = compute_masked_accuracy(student_preds, last_targets, last_loss_mask)

            if ema_student is not None:
                ema_student_module = ema_student
                with torch.no_grad(), ctx:
                    ema_student_logits, ema_student_loss = ema_student_module(
                        last_inputs,
                        last_targets,
                        loss_mask=eval_loss_mask,
                    )
                if ema_student_loss is not None:
                    ema_student_loss_logged = ema_student_loss.item()
                ema_student_preds = ema_student_logits.argmax(dim=-1)
                ema_student_acc_logged = compute_masked_accuracy(ema_student_preds, last_targets, last_loss_mask)

        # Logging
        if not teacher_blocked:
            train_loss_sum += loss.item() * A
            if not math.isnan(loss_after):
                train_loss_after_sum += loss_after
            elapsed += 1
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if (not teacher_blocked) and (current_teacher_step in log_iter_set) and master_process:
            avg_loss = train_loss_sum / elapsed if elapsed else float('nan')
            avg_loss_after = train_loss_after_sum / elapsed if elapsed else float('nan')
            with torch.no_grad():
                features = model.forward_features(x0['inputs'])
            current_logs = {
                "iter": current_teacher_step,
                "compute": 6 * P * tokens_per_iter * current_teacher_step,
                "tokens": current_teacher_step * tokens_per_iter,
                "train_loss": avg_loss,
                "train_loss_after": avg_loss_after,
                'h': torch.sqrt(torch.mean(torch.square(features))).item(),
                "lr": lr_schedule * lr,
                "mfu": running_mfu * 100,
                "student_loss": student_loss_logged,
                "distill_kl": distill_kl_logged,
                "teacher_acc": teacher_acc_logged,
                "student_acc": student_acc_logged,
                "K_req": cumulative_k_req,
                "student_tokens": cumulative_student_tokens,
                "student_compute": 6 * student_param_count * cumulative_student_tokens,
                "ema_train_loss": ema_train_loss_logged,
                "ema_student_loss": ema_student_loss_logged,
                "ema_teacher_acc": ema_teacher_acc_logged,
                "ema_student_acc": ema_student_acc_logged,
            }
            if eval_callback is not None:
                extra_eval = eval_callback(raw_model, meta, device=device)
                if extra_eval:
                    current_logs.update(extra_eval)
            if prev_features is not None:
                dh = torch.sqrt(torch.mean(torch.square(features - prev_features)))
                current_logs['dh'] = dh.item()
            prev_features = features
            for k, v in current_logs.items():
                if k in log_df:
                    log_df[k].append(v)
            if elapsed:
                Ls = np.array(log_df['train_loss']) # (iter,)
                Ts = np.array(log_df['tokens']) # (iter,)
                L = current_logs['train_loss']
                dLs = Ls - L
                auc = np.trapz(dLs, Ts)
                current_logs['K_auc'] = auc
                log_df['K_auc'].append(auc)
            if wandb_log:
                wandb.log(current_logs)
            if elapsed:
                pbar.set_description(f"L: {avg_loss:.3f}")
            train_loss_sum = 0
            train_loss_after_sum = 0
            elapsed = 0
        if not teacher_blocked:
            teacher_iter += 1
            pbar.update(1)
        loop_iter += 1
    if wandb_log and master_process:
        wandb.finish()
    if ddp:
        destroy_process_group()
    df = pd.DataFrame(log_df)
    if return_model:
        models = {'teacher': raw_model}
        if requential and student is not None:
            models['student'] = student.module if ddp else student
        if ema_teacher is not None:
            models['ema_teacher'] = ema_teacher
        if ema_student is not None:
            models['ema_student'] = ema_student
        return df, models
    return df

def get_lr_schedule(t, warmup, schedule, T):
    if t < warmup:
        return t / warmup
    if schedule == 'const':
        return 1.0
    elif schedule == 'linear':
        return 1.0 - (t - warmup) / (T - warmup)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

if __name__ == "__main__":
    fire.Fire(train)
