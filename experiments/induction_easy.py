import torch
import torch.nn.functional as F
from soph.train import train
from soph.datasets.markov import NGrams
from soph.utils.config_generator import FixedNumpySeed
import numpy as np
from soph.utils.rng import sha_hash

def predict_half_forward_wmatrix():
    def fn(self, inputs, targets=None, loss_mask=None):
        if isinstance(inputs, dict):
            idx = inputs['tokens']
            logitsvec = inputs['logitsvec']
        else:
            idx = inputs
            logitsvec = None

        device = idx.device
        s = idx.size(1)
        pos = torch.arange(0, s, dtype=torch.long, device=device)
        token_ids = torch.where(idx >= 0, idx, torch.zeros_like(idx))
        tok_emb = self.transformer.wte(token_ids)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        prefix_len = 0
        if logitsvec is not None:
            if logitsvec.dim() != 3:
                raise ValueError("logitsvec must have shape (batch, rows, vocab)")
            logitsvec = logitsvec.to(x.dtype)
            nrows_shown = logitsvec.shape[-2]
            V = logitsvec.shape[-1]
            if nrows_shown + 1 > x.size(1):
                raise ValueError("logitsvec has more rows than available prefix tokens")
            x[:, 1:nrows_shown + 1, :V] = logitsvec
            prefix_len = nrows_shown + 1

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must match the shape of input tokens")
            ce = F.cross_entropy(
                logits[:, prefix_len:, :].permute(0, 2, 1),
                targets[:, prefix_len:],
                ignore_index=-1,
                reduction='none',
            )
            if loss_mask is not None:
                mask_slice = loss_mask[:, prefix_len:].to(ce.dtype)
                loss = (ce * mask_slice).sum() / mask_slice.sum().clamp(min=1)
            else:
                loss = ce.mean()
        return logits, loss
    return fn

def generate_markov_synthetic():
    @torch.no_grad()
    def fn(self, raw_batch, temperature=1.0, **generate_kwargs):
        if not isinstance(raw_batch, dict):
            raise ValueError("raw_batch must contain 'data' and 'logitsvec'")
        tokens = raw_batch['data']
        logitsvec = raw_batch['logitsvec']
        if tokens.dim() != 2:
            raise ValueError("raw data tensor must be 2D")
        if logitsvec.dim() != 3:
            raise ValueError("logitsvec tensor must be 3D")

        total_len = tokens.size(1)
        prefix_len = logitsvec.size(1) + 1
        generated = tokens[:, :prefix_len].clone()
        sample_len = max(total_len - prefix_len, 0)

        def forward_with_cache(token_chunk, override_logitsvec, past_key_values):
            num_layers = len(self.transformer.h)
            if past_key_values is None:
                past_key_values = [None] * num_layers
            else:
                if len(past_key_values) != num_layers:
                    raise ValueError("past_key_values must match number of transformer layers")
                past_key_values = list(past_key_values)

            past_length = 0
            if past_key_values[0] is not None:
                past_length = past_key_values[0][0].size(-2)

            device = token_chunk.device
            s = token_chunk.size(1)
            pos = torch.arange(past_length, past_length + s, dtype=torch.long, device=device)
            safe_tokens = torch.where(token_chunk >= 0, token_chunk, torch.zeros_like(token_chunk))
            tok_emb = self.transformer.wte(safe_tokens)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            if override_logitsvec is not None:
                logits_override = override_logitsvec.to(x.dtype)
                nrows_shown = logits_override.size(1)
                vocab = logits_override.size(2)
                x[:, 1:nrows_shown + 1, :vocab] = logits_override

            new_past = []
            for layer_idx, block in enumerate(self.transformer.h):
                layer_past = past_key_values[layer_idx]
                x, present = block(x, past_key_value=layer_past, use_cache=True)
                new_past.append(present)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            return logits, new_past

        logits, past_key_values = forward_with_cache(generated, logitsvec, None)
        next_logits = logits[:, -1, :]

        for _ in range(sample_len):
            if temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-5)
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            logits, past_key_values = forward_with_cache(next_token, None, past_key_values)
            next_logits = logits[:, -1, :]

        if generated.size(1) > total_len:
            generated = generated[:, :total_len]
        elif generated.size(1) < total_len:
            generated = torch.cat([generated, tokens[:, generated.size(1):total_len]], dim=1)

        teacher_inputs = {
            'tokens': generated[:, :-1],
            'logitsvec': logitsvec,
        }
        teacher_targets = generated[:, 1:]
        teacher_logits, _ = self(teacher_inputs, teacher_targets)

        mask = torch.zeros_like(teacher_targets, dtype=torch.bool)
        if prefix_len < mask.size(1):
            mask[:, prefix_len:] = True

        return {
            'data': generated,
            'logitsvec': logitsvec,
            'logits': teacher_logits,
            'mask': mask,
        }

    return fn

def forward_features_wrapper(original_forward_features):
    """Wrapper for forward_features that handles dict inputs."""
    def fn(self, inputs, past_key_values=None, use_cache=False):
        if isinstance(inputs, dict):
            idx = inputs['tokens']
        else:
            idx = inputs
        return original_forward_features(self, idx, past_key_values, use_cache)
    return fn

def patch_predict_half_wmatrix():
    newforward = predict_half_forward_wmatrix()
    new_generate = generate_markov_synthetic()
    def patch(model):
        print("patching forward method for markov hidden task")
        original_forward_features = model.__class__.forward_features
        model.__class__.forward = newforward
        model.__class__.forward_features = forward_features_wrapper(original_forward_features)
        model.__class__.generate_synthetic = new_generate
    return patch


def fixed_params_datagen(ngrams=2, width=128, symbols=2, batch_size=1000, hidden_params=0, seed=42):
    """Generate Markov chain data where hidden_params rows of the transition matrix are hidden."""
    num_visible_rows = symbols**(ngrams-1) - hidden_params
    meta = {'ngrams': ngrams, 'symbols': symbols,
     'vocab_size': symbols+1, 'hidden_params': hidden_params, 'seq_len': width+1+num_visible_rows*ngrams}
    with FixedNumpySeed(seed):
        dataset = NGrams(ngrams,width,symbols,free_params=-1)

    row_ids = np.random.permutation(symbols**(ngrams-1))[:num_visible_rows]
    prefixes = row_ids[:,None]
    prefixes = prefixes.astype(np.int64)
    prefixes = prefixes.reshape(-1)
    prefixes = np.insert(prefixes,0,symbols)
    prefixes = prefixes[None]+np.zeros((batch_size,1),dtype=np.int64)
    def get_batch(iteration,rank=0):
        key = sha_hash((iteration,seed,rank))
        out, transitions = dataset.generate(batch_size,key,matrix=True)
        logitvecs = np.log(transitions[:,row_ids,:]).astype(np.float32)
        index_hints = prefixes
        conc_out = np.concatenate([index_hints, out],1)
        loss_mask = np.zeros_like(conc_out, dtype=np.uint8)
        loss_mask[:, num_visible_rows+1:] = 1
        return {
            'data': conc_out,
            'logitsvec': logitvecs,
            'loss_mask': loss_mask,
        }

    return get_batch, meta, dataset.avg_ngram_losses(row_ids)

def transform_batch(batch):
    data_np = batch['data']
    logitsvec_np = batch['logitsvec']
    loss_mask_np = batch['loss_mask']

    data_tensor = torch.from_numpy(data_np.astype(np.int64))
    logits_tensor = torch.from_numpy(logitsvec_np.astype(np.float32))

    inputs = {
        'tokens': data_tensor[:, :-1],
        'logitsvec': logits_tensor,
    }
    targets = data_tensor[:, 1:]
    loss_mask = torch.from_numpy(loss_mask_np[:, 1:].astype(np.bool_))
    raw = {
        'data': data_tensor,
        'logitsvec': logits_tensor,
    }
    return {
        'inputs': inputs,
        'targets': targets,
        'loss_mask': loss_mask,
        'raw': raw,
    }

def trial(cfg):
    data_cfg = cfg.pop('data_cfg')
    cfg['get_batch'],meta, ngram_losses = fixed_params_datagen(**data_cfg)
    cfg['logged_kwargs'] = data_cfg
    for n,v in ngram_losses.items():
        cfg['logged_kwargs'][f'gram_loss_{n}']=v
    cfg['meta']=meta
    cfg['apply_patch']=patch_predict_half_wmatrix()
    cfg['transform_batch']=transform_batch
    train(**cfg)
    print('finished run')


if __name__ == "__main__":
    import copy
    from soph.utils.config_generator import dispatch_multigpu, grid_iter

    seed_values = [0]
    width_values = [512]
    data_spec = {'ngrams': 2, 'width': width_values[0], 'symbols': 8, 'batch_size':lambda cfg: cfg['B'],
        'hidden_params': [0, 2, 4, 6, 8], 'seed': 0}

    cfg_spec = copy.deepcopy(train.__kwdefaults__)

    test=False

    cfg_spec.update({
        'arch': 'gpt',
        'L': [3],
        'D': [128],
        'A': 1,
        'T': 3000,
        'warmup': 15,
        'tag': 'markov_hidden_kl_capped_sweep_final',
        'compile': False,
        'wandb_log': not test,
        'wandb_project': 'requential',
        'B': 384,
        'data_cfg': data_spec,
        'log_history': False,
        'student_speed': 1,
        'max_kl': [0.005],
        'ema_steps': 50,
        'requential': True,
    })
    if test:
        example = next(iter(grid_iter(cfg_spec)))
        trial(example)
    else:
        for seed in seed_values:
            for width in width_values:
                cfg_seed = copy.deepcopy(cfg_spec)
                cfg_seed['data_cfg']['seed'] = seed
                cfg_seed['data_cfg']['width'] = width
                print(f"=== Running seed {seed}, width {width} ===")
                dispatch_multigpu(trial, cfg_seed, num_trials=-1, ordered=True)
