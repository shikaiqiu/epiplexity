import torch
import torch.nn.functional as F
from soph.train_old import train_basic
from functools import partial
import numpy as np
from soph.utils.rng import sha_hash

def predict_half_forward_wmaskedbits():
    
    def fn(self, idx, targets=None):
        # nbits specifies number of leading hidden bits in the input
        s0 = idx.shape[1]
        # idx = idx[:,nbits:]
        # if targets is not None:
        #     targets = targets[:,nbits:]
        device = idx.device
        b, s = idx.size()
        assert s <= self.config.S, f"Cannot forward sequence of length {t}, block size is only {self.config.S}"
        pos = torch.arange(0, s, dtype=torch.long, device=device) # shape (s)

        # forward the GPT model itself
        #assert False
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, s, D)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (s, D)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # B, S, V
            # CHANGED LINES
            loss = F.cross_entropy(logits.permute((0,2,1)), targets,ignore_index=-1,reduction='none')
            loss = loss.mean()#loss[:,(s0-1)//2:].mean()
            # suppose W=100
            # loss of shape (B,199)
            # loss[:,99:]
            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    return fn

def patch_predict_half_wobits():
    newforward = predict_half_forward_wmaskedbits()
    def patch(model):
        print("patching forward method of the model \
            to predict only the second half, and mask the first bits")
        model.__class__.forward = newforward
    return patch

from soph.datasets.CA import generate

def ca_datagen(steps=4, width=32, rule=30, batch_size=1000, burnin=False, seed=42,reversed=False):
    meta = {'steps': steps, 'width': width, 'rule': rule,
     'vocab_size': 3, 'seq_len': 2 * width, 'reversed':reversed}
     
    sep=False
    generate_wargs = partial(generate, batch_size, steps, width, rule, burnin, sep)
    
    def get_batch(iteration,rank=0):
        key = sha_hash((iteration,seed,rank))
        out = generate_wargs(key)
        if reversed:
            out = np.concatenate((out[...,width:],out[...,:width]),axis=-1)
        return np.insert(out,0,2,-1).astype(np.int32)
    
    return get_batch, meta

def run(cfg):
    cfg['get_batch'],cfg['meta'] = ca_datagen(batch_size=cfg['B'],seed=cfg['seed'],
    reversed=cfg['reversed'],steps=cfg['rounds'],width=cfg['width'])
    cfg['apply_patch'] = patch_predict_half_wobits()
    train_basic(**cfg)

if __name__ == "__main__":
    import copy
    from soph.utils.config_generator import dispatch_multigpu, grid_iter

    debug = False

    cfg = copy.deepcopy(train_basic.__kwdefaults__)
    cfg.update({
        'L': 8,
        'D': lambda cfg: 128 if cfg['width']<=32 else 192,
        'A': 1,
        'T': lambda cfg: 50000 if (cfg['width']<=32 and not cfg['reversed']) else 100000,
        'B': 4*128,
        'warmup': 500,
        'tag': 'eca30_multiple_sizes',#''sbox_shifts_vrounds',
        'compile': False,
        'wandb_log': not debug,
        'wandb_project': 'soi',
        'log_history':False,
        'width':[16,24,32,48,64],
        'reversed':[True,False],
        'seed':[68,37],
        'rounds':8,
        #[16,24,32,48,64],
        #'device':'cpu',
    })
    if debug:
        nextcfg = list(grid_iter(cfg))[0]
        run(nextcfg)
    else:
        dispatch_multigpu(run,cfg,ordered=True)