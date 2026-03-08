
import os
import torch
import torch.nn.functional as F
from soph.train import train
import copy
from soph.datasets.CA import generate
import numpy as np
from soph.utils.rng import sha_hash

def predict_half_forward(self, idx, targets=None, loss_mask=None):
    """Forward pass that only trains on the latter half of the sequence."""
    if idx.dim() != 2:
        raise ValueError("idx must have shape (batch, seq_len)")
    s = idx.size(1)
    if s > self.config.S:
        raise ValueError(f"Cannot forward sequence of length {s}, block size is only {self.config.S}")

    x = self.forward_features(idx, use_cache=False)
    x = self.transformer.ln_f(x)

    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.permute((0, 2, 1)),
            targets,
            ignore_index=-1,
            reduction='none',
        )
        loss_slice = loss[:, (s - 1) // 2:]
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must match targets shape")
            mask_slice = loss_mask[:, (s - 1) // 2:].to(loss.dtype)
            loss = (loss_slice * mask_slice).sum() / mask_slice.sum().clamp(min=1)
        else:
            loss = loss_slice.mean()
    else:
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss

def generate_synthetic(self, idx, temperature=1.0, **generate_kwargs):
    if idx.dim() != 2:
        raise ValueError("idx must have shape (batch, seq_len)")
    total_len = idx.size(1)
    if total_len == 0:
        raise ValueError("sequence must contain tokens")

    input_len = total_len // 2
    target_len = total_len - input_len
    if input_len == 0 or target_len == 0:
        raise ValueError("sequence must contain both input and target tokens")

    prefix = idx[:, :input_len]
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
    teacher_logits[:, input_len - 1:, :] = logits
    mask = torch.zeros(idx.size(0), pred_len, dtype=torch.bool, device=device)
    mask[:, input_len - 1:] = True

    synthetic_full = generated
    return {
        'data': synthetic_full,
        'logits': teacher_logits,
        'mask': mask,
    }

def patch_predict_half(model):
    print("patching forward method to focus on the second half and enabling target generation")
    model.__class__.forward = predict_half_forward
    model.__class__.generate_synthetic = generate_synthetic


def _masked_accuracy(preds, targets, mask):
    if mask is None:
        return float('nan')
    denom = mask.sum().item()
    if denom == 0:
        return float('nan')
    correct = (preds == targets) & mask
    return (correct.sum().float() / denom).item()


def ca_datagen(steps=1, width=128, rule=30, batch_size=1000, burnin=True, seed=42):
    """Generate CA data for final-state-only prediction (input -> output after N steps).

    If rule is a list, samples a random rule for each batch.
    """
    seq_len = 2 * width  # input width + output width
    target_len = width

    # Handle list of rules
    rules = rule if isinstance(rule, (list, tuple, np.ndarray)) else [rule]
    n_rules = len(rules)

    meta = {
        'steps': steps,
        'width': width,
        'rule': rule,
        'vocab_size': 2,
        'seq_len': seq_len,
        'target_len': target_len,
        'burnin': burnin,
    }

    def get_batch(iteration, rank=0):
        key = sha_hash((iteration, seed, rank))
        rng = np.random.default_rng(key)
        selected_rule = int(rules[rng.integers(0, n_rules)])
        samples = generate(batch_size, steps, width, selected_rule, burnin, False, key)

        loss_mask = np.zeros_like(samples, dtype=np.uint8)
        loss_mask[:, width:] = 1

        return {'data': samples, 'loss_mask': loss_mask}

    return get_batch, meta

def run(cfg, *, return_model=False, save_model=True):
    data_cfg = cfg.pop('data_cfg')
    cfg.pop('return_model', None)
    cfg.pop('save_model', None)
    default_out = cfg.get('out_dir', 'out')
    if default_out == 'out':
        rule_tag = data_cfg['rule'] if isinstance(data_cfg.get('rule'), (int, np.integer)) else 'mixed'
        cfg['out_dir'] = os.path.join('out', f"trainonca_rule_{rule_tag}")
    cfg['get_batch'], cfg['meta'] = ca_datagen(batch_size=cfg['B'], **data_cfg)

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
