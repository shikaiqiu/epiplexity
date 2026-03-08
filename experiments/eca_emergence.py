
import os
import torch
from soph.train import train
from functools import partial
import copy
from soph.datasets.CA import generate, generate_all_steps, generate_freq_steps
from soph.utils.config_generator import FixedNumpySeed
import numpy as np
from soph.utils.rng import sha_hash


def _masked_accuracy(preds, targets, mask):
    if mask is None:
        return float('nan')
    denom = mask.sum().item()
    if denom == 0:
        return float('nan')
    correct = (preds == targets) & mask
    return (correct.sum().float() / denom).item()


def make_synthetic_patcher(target_start, target_len):
    import types

    def generate_synthetic(self, idx, temperature=1.0, **generate_kwargs):
        if idx.dim() != 2:
            raise ValueError("idx must have shape (batch, seq_len)")
        total_len = idx.size(1)
        if total_len <= target_start:
            raise ValueError("sequence must contain prefix and targets")
        prefix = idx[:, :target_start]
        generated, logits = self.generate(
            prefix,
            tokens_to_generate=target_len,
            temperature=temperature,
            return_logits=True,
            **generate_kwargs,
        )
        device = idx.device
        vocab_size = logits.size(-1)
        pred_len = total_len - 1
        teacher_logits = torch.zeros(idx.size(0), pred_len, vocab_size, device=device, dtype=logits.dtype)
        teacher_logits[:, target_start - 1:, :] = logits
        mask = torch.zeros(idx.size(0), pred_len, dtype=torch.bool, device=device)
        mask[:, target_start - 1:] = True
        return {
            'data': generated,
            'logits': teacher_logits,
            'mask': mask,
        }

    def patch_fn(model):
        model.generate_synthetic = types.MethodType(generate_synthetic, model)

    return patch_fn

def ca_datagen(steps=1, width=128, rule=30, batch_size=1000, burnin=True, seed=42, predict_freq=1):
    sep = False
    sep_len = int(sep) if isinstance(sep, (bool, np.bool_)) else int(sep)
    steps_max = int(np.max(steps)) if isinstance(steps, (list, tuple, np.ndarray)) else int(steps)
    predict_freq = int(predict_freq)
    if predict_freq <= 1:
        seq_len = width + sep_len + steps_max * width
        generate_fn = partial(generate_all_steps, batch_size, steps_max, width, rule, burnin, sep)
    elif predict_freq >= steps_max:
        seq_len = 2 * width + sep_len
        generate_fn = partial(generate, batch_size, steps_max, width, rule, burnin, sep)
    else:
        target_steps = list(range(predict_freq, steps_max + 1, predict_freq))
        if target_steps[-1] != steps_max:
            target_steps.append(steps_max)
        n_targets = len(target_steps)
        seq_len = width + sep_len + n_targets * width
        generate_fn = partial(generate_freq_steps, batch_size, steps_max, width, rule, burnin, sep, predict_freq)

    meta = {
        'steps': steps,
        'width': width,
        'rule': rule,
        'vocab_size': 2,
        'seq_len': seq_len,
        'target_len': seq_len - width - sep_len,
        'burnin': burnin,
        'predict_freq': predict_freq,
        'sep': sep,
    }

    def get_batch(iteration, rank=0):
        key = sha_hash((iteration, seed, rank))
        out = generate_fn(key)

        samples = out

        loss_mask = np.zeros_like(samples, dtype=np.uint8)
        target_start = width + sep_len
        loss_mask[:, target_start:] = 1

        batch = {'data': samples, 'loss_mask': loss_mask}
        return batch

    return get_batch, meta

def run(cfg, *, return_model=False, save_model=True):
    data_cfg = cfg.pop('data_cfg')
    cfg.pop('return_model', None)
    cfg.pop('save_model', None)
    default_out = cfg.get('out_dir', 'out')
    if default_out == 'out':
        rule_tag = data_cfg['rule'] if isinstance(data_cfg.get('rule'), (int, np.integer)) else 'mixed'
        cfg['out_dir'] = os.path.join('out', f"trainonca_rule_{rule_tag}")
    cfg['get_batch'],cfg['meta'] = ca_datagen(batch_size=cfg['B'],**data_cfg)
    if cfg['B_tokens'] is not None:
        cfg['B'] = cfg['B_tokens'] // cfg['meta']['target_len']
        print(f"Using B_tokens={cfg['B_tokens']} => B={cfg['B']}")
        cfg['get_batch'],cfg['meta'] = ca_datagen(batch_size=cfg['B'],**data_cfg)
    steps_max = int(np.max(cfg['meta']['steps'])) if isinstance(cfg['meta']['steps'], (list, tuple, np.ndarray)) else int(cfg['meta']['steps'])
    freq = cfg['meta']['predict_freq']
    target_start = cfg['meta']['width'] + int(cfg['meta']['sep'])
    target_len = cfg['meta']['target_len']

    user_patch = cfg.get('apply_patch')
    synth_patch = make_synthetic_patcher(target_start, target_len)

    if user_patch is None:
        cfg['apply_patch'] = synth_patch
    else:
        def composed_patch(model):
            synth_patch(model)
            user_patch(model)
        cfg['apply_patch'] = composed_patch

    want_model = return_model or save_model
    if want_model:
        log_df, models = train(return_model=True, **cfg)
    else:
        log_df = train(return_model=False, **cfg)
        models = None

    if save_model and models is not None:
        out_dir = cfg.get('out_dir', 'out')
        os.makedirs(out_dir, exist_ok=True)
        log_dict = log_df.to_dict() if hasattr(log_df, 'to_dict') else log_df
        cfg_to_save = {}
        for k, v in cfg.items():
            if k == 'get_batch' or callable(v):
                continue
            cfg_to_save[k] = copy.deepcopy(v)
        teacher_model = models.get('teacher')
        student_model = models.get('student')
        payload = {'model_state_dict': teacher_model.state_dict() if teacher_model is not None else None,
                   'student_state_dict': student_model.state_dict() if student_model is not None else None,
                   'config': cfg_to_save,
                   'data_cfg': data_cfg,
                   'log_df': log_dict}
        torch.save(payload, os.path.join(out_dir, 'model_final.pt'))

    if return_model:
        return log_df, models
    return log_df


if __name__ == "__main__":
    import copy
    from soph.utils.config_generator import dispatch_multigpu, grid_iter

    debug = False

    rules = [0, 32, 160, 232, # class 1: convergent
            4, 15, 108, 218, # class 2: periodic
            22, 30, 126, 150, # class 3: chaotic
            41, 54, 106, 110] # class 4: complex

    subset = [0, 32, 4, 15, 22, 30, 41, 54, 106, 110]

    cfg = copy.deepcopy(train.__kwdefaults__)

    width = 64
    cfg.update({
        'arch': 'gpt',
        'L': [1, 2, 4, 8, 16, 32],
        'D': [16, 32, 64, 128],
        'd_head': 32,
        'B_tokens': 96*4*64*6,
        'T': 1e9 // (96*4*64*6),
        'A': 1,
        'lr': 0.06,
        'warmup': 100,
        'tag': 'eca_emergence',
        'compile': False,
        'wandb_log': not debug,
        'wandb_project': 'requential',
        'log_geometric': True,
        'requential': True,
        'student_speed': 1,
        'ema_steps': 50,
        'data_cfg': {'steps': [64],'rule': [54], 'width': width, 'predict_freq': [64, 32, 16, 8, 4]},
        'apply_patch': None,
    })

    if debug:
        nextcfg = list(grid_iter(cfg))[0]
        run(nextcfg)
    else:
        dispatch_multigpu(run,cfg,ordered=True)
