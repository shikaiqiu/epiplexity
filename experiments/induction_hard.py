import torch
import torch.nn.functional as F
from soph.train import train
from functools import partial
import types
from soph.datasets.CA import generate
from soph.utils.config_generator import FixedNumpySeed
import numpy as np
from soph.utils.rng import sha_hash

def predict_half_forward_wmaskedbits(nbits=0):

    def fn(self, idx, targets=None, loss_mask=None):
        total_len = idx.shape[1]
        device = idx.device
        b, s = idx.size()
        assert s <= self.config.S, f"Cannot forward sequence of length {s}, block size is only {self.config.S}"

        x = self.forward_features(idx, use_cache=False)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.permute((0, 2, 1)), targets, ignore_index=-1, reduction='none')
            base_start = (total_len + nbits - 1) // 2
            pred_start = base_start - nbits
            loss_slice = loss[:, pred_start:]
            if loss_mask is not None:
                mask_slice = loss_mask[:, pred_start:].to(loss.dtype)
                loss = (loss_slice * mask_slice).sum() / mask_slice.sum().clamp(min=1)
            else:
                loss = loss_slice.mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    return fn

def generate_hiddenbits_synthetic(self, idx, temperature=1.0, **generate_kwargs):
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


def patch_predict_half_wobits(nbits=0):
    newforward = predict_half_forward_wmaskedbits(nbits=nbits)
    def patch(model):
        print("patching forward method of the model to predict only the second half, and mask the first bits")
        model.__class__.forward = newforward
        model.__class__.generate_synthetic = generate_hiddenbits_synthetic
    return patch

def ca_datagen(steps=1, width=128, rule=30, batch_size=1000, burnin=True, seed=42, nbits=0):
    """Generate CA data with hidden bits - first nbits of input are masked."""
    assert 0 <= nbits < width, "nbits must be between 0 and width-1"
    input_len = width - nbits
    target_len = width
    seq_len = input_len + target_len
    meta = {
        'steps': steps,
        'width': width,
        'rule': rule,
        'nbits': nbits,
        'vocab_size': 2,
        'seq_len': seq_len,
        'target_len': target_len,
        'burnin': burnin,
    }

    def get_batch(iteration, rank=0):
        key = sha_hash((iteration, seed, rank))
        out = generate(batch_size, steps, width, rule, burnin, False, key)
        inputs = out[:, :width]
        outputs = out[:, width:]
        visible_inputs = inputs[:, nbits:] if nbits > 0 else inputs
        combined = np.concatenate((visible_inputs, outputs), axis=-1)
        loss_mask = np.zeros_like(combined, dtype=np.uint8)
        loss_mask[:, visible_inputs.shape[1]:] = 1
        return {'data': combined, 'loss_mask': loss_mask}

    return get_batch, meta

def run(cfg):
    data_cfg = cfg.pop('data_cfg')
    cfg['get_batch'],cfg['meta'] = ca_datagen(batch_size=cfg['B'],**data_cfg)
    cfg['apply_patch'] = patch_predict_half_wobits(nbits=cfg['meta']['nbits'])
    train(**cfg)

if __name__ == "__main__":
    import copy
    from soph.utils.config_generator import dispatch_multigpu, grid_iter
    cfg = copy.deepcopy(train.__kwdefaults__)
    debug = False

    cfg.update({
        'L': [3],
        'D': 256,
        'A': 1,
        'T': 20000,
        'B': 96*4 * 128 // 32,
        'warmup': 100,
        'tag': 'ca_hidden',
        'compile': False,
        'wandb_log': not debug,
        'wandb_project': 'requential',
        'log_geometric': True,
        'data_cfg': {'steps': 4,'rule': 30, 'width': 32,'nbits': [0, 1, 2, 3, 4, 5]},
        'student_speed': 1,
        'max_kl': [0.03],
        'ema_steps': 100,
        'requential': True,
    })
    if debug:
        nextcfg = list(grid_iter(cfg))[0]
        run(nextcfg)
    else:
        dispatch_multigpu(run,cfg,ordered=True)
