"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


def mparams2gptcfg(params_millions: float) -> dict:
    # Use power law fits
    n_layers = int(round(0.77 * params_millions**0.5955))
    n_heads = int(round(2.84 * params_millions**0.2753))
    d_head = int(round(11.78 * params_millions**0.3812))
    D = n_heads * d_head

    return {
        'L': n_layers,
        'D': D,
        'd_head': d_head
        # other fields like S, V, dropout remain fixed in GPTConfig or can be added if needed
    }

@dataclass
class GPTConfig:
    S: int = 1024 # Seq length
    V: int = 50304 # GPT-2 V of 50257, padded up to nearest multiple of 64 for efficiency
    L: int = 12
    D: int = 768
    d_head: int = 64 # keep at 64 to simplicity
    dropout: float = 0.0
    qk_norm: bool = True
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    block_exponent: float = 1.0 # block multiplier is L ** {-block_exponent}
    qk_exponent: float = 1.0 # scale qk by d_head ** qk_exponent
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.D % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.D, 3 * config.D, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.D, config.D, bias=config.bias)
        # regularization
        self.q_ln = LayerNorm(config.d_head, bias=config.bias) if config.qk_norm else nn.Identity()
        self.k_ln = LayerNorm(config.d_head, bias=config.bias) if config.qk_norm else nn.Identity()
        if config.qk_exponent == 0.5:
            self.qk_scale = 1
        elif config.qk_exponent == 1:
            self.qk_scale = 8 / (config.d_head ** 0.5)
        else:
            raise ValueError("qk_exponent must be 0.5 or 1")
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.D = config.D
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.S, config.S))
                                        .view(1, 1, config.S, config.S))
    def forward(self, x, past_key_value=None, use_cache=False):
        B, S, D = x.size() # batch size, sequence length, embedding dimensionality (D)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.D, dim=2)
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2) # (B, nh, S, dh)
        q = self.q_ln(q)
        k = k.view(B, S, self.n_head, self.d_head).transpose(1, 2) # (B, nh, S, dh)
        k = self.k_ln(k) * self.qk_scale
        v = v.view(B, S, self.n_head, self.d_head).transpose(1, 2) # (B, nh, S, dh)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            past_len = past_k.size(-2)

        total_len = k.size(-2)
        device = q.device

        def manual_attention():
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            key_positions = torch.arange(total_len, device=device)
            query_positions = past_len + torch.arange(S, device=device)
            causal_mask = query_positions.view(1, 1, S, 1) >= key_positions.view(1, 1, 1, total_len)
            att = att.masked_fill(~causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            return att @ v

        # causal self-attention; Self-attend: (B, nh, S, dh) x (B, nh, dh, S_total) -> (B, nh, S, S_total)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if past_len == 0:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                query_positions = past_len + torch.arange(S, device=device)
                key_positions = torch.arange(total_len, device=device)
                allow = query_positions.unsqueeze(-1) >= key_positions.unsqueeze(0)
                mask = torch.zeros((S, total_len), device=device, dtype=q.dtype)
                mask.masked_fill_(~allow, float('-inf'))
                mask = mask.unsqueeze(0).unsqueeze(0)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )
        else:
            # manual implementation of attention
            y = manual_attention() # (B, nh, S, S_total) x (B, nh, S_total, dh) -> (B, nh, S, dh)
        y = y.transpose(1, 2).contiguous().view(B, S, D) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if use_cache:
            return y, (k, v)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.D, 4 * config.D, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.D, config.D, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.D, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.D, bias=config.bias)
        self.mlp = MLP(config)
        self.res_scale = config.L ** config.block_exponent

    def forward(self, x, past_key_value=None, use_cache=False):
        attn_input = self.ln_1(x)
        attn_out = self.attn(attn_input, past_key_value=past_key_value, use_cache=use_cache)
        if use_cache:
            attn_out, present = attn_out
        else:
            present = None
        x = x + attn_out / self.res_scale
        x = x + self.mlp(self.ln_2(x)) / self.res_scale
        if use_cache:
            return x, present
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.V is not None
        assert config.S is not None
        self.config = config
        if config.d_head > config.D:
            config.d_head = config.D
            print(f"WARNING: d_head ({config.d_head}) is greater than D ({config.D}), setting d_head to D")
        self.config.n_head = config.D / config.d_head
        assert self.config.n_head.is_integer(), "d_head must divide D"
        self.config.n_head = int(self.config.n_head)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.V, config.D),
            wpe = nn.Embedding(config.S, config.D),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.L)]),
            ln_f = LayerNorm(config.D, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.D, config.V, bias=False)

        # init all weights as O(1/sqrt(fan_in))
        # except embeddings which are as O(1)
        self.apply(self._init_weights)
        # zero init (inf -> 1) layers
        self.lm_head.weight.data.zero_()
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # this includes W_down and W_o
                torch.nn.init.zeros_(p)

        # report number of parameters
        print("number of parameters: %.4fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(module.weight.size(1)))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_features(self, idx, past_key_values=None, use_cache=False):
        """Return hidden features prior to the final LayerNorm.

        When ``use_cache`` is True, also returns the updated ``past_key_values``.
        """
        device = idx.device
        b, s = idx.size()
        if s == 0:
            raise ValueError("input sequence length must be > 0")

        num_layers = self.config.L
        if past_key_values is None:
            past_key_values = [None] * num_layers
        else:
            if len(past_key_values) != num_layers:
                raise ValueError("past_key_values must match the number of transformer layers")
            past_key_values = list(past_key_values)

        past_length = 0
        if past_key_values and past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(-2)

        total_length = past_length + s
        if total_length > self.config.S:
            raise ValueError(
                f"Sequence length {total_length} exceeds block size {self.config.S}"
            )

        pos = torch.arange(past_length, past_length + s, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        new_past = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if use_cache else None
            if use_cache:
                x, present = block(x, past_key_value=layer_past, use_cache=True)
                new_past.append(present)
            else:
                x = block(x, past_key_value=None, use_cache=False)

        if use_cache:
            return x, new_past
        return x

    def _forward_with_past(self, idx, past_key_values=None, use_cache=False):
        if use_cache:
            x, new_past = self.forward_features(
                idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            x = self.forward_features(
                idx,
                past_key_values=past_key_values,
                use_cache=False,
            )
            new_past = None

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_past
        return logits

    def forward(self, idx, targets=None, loss_mask=None):
        logits = self._forward_with_past(idx, use_cache=False)

        if targets is not None:
            loss = F.cross_entropy(
                logits.permute((0, 2, 1)),
                targets,
                ignore_index=-1,
                reduction='none'
            )
            if loss_mask is not None:
                if loss_mask.shape != targets.shape:
                    raise ValueError("loss_mask must match targets shape")
                mask = loss_mask.to(loss.dtype)
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()
        else:
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, vec_lr=6e-4):
        """
        Configure optimizer with µP learning rate scaling for transformers.
        Args:
            weight_decay (float): Weight decay coefficient
            learning_rate (float): Base learning rate
            betas (tuple): Adam beta parameters
            device_type (str): Device type ('cuda' or 'cpu')
            vec_lr (float): Learning rate for embedding and LayerNorm parameters.
                                    Defaults to value for GPT-2 Small
        """

        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Initialize parameter groups
        param_groups = []
        peak_lrs = {}  # Store base learning rates for monitoring
        
        for name, param in param_dict.items():
            # Initialize group settings
            group = {
                'params': param,
                'weight_decay': weight_decay if param.dim() >= 2 else 0,
                'name': name
            }
            
            # Determine learning rate and weight decay for each parameter type
            if 'wte.' in name or 'wpe.' in name or 'ln' in name or 'time_emb' in name:
                # Embeddings and LayerNorms use vec_lr
                group['lr'] = vec_lr
            elif 'bias' in name:
                # Biases use vec_lr and no weight decay
                group['lr'] = vec_lr
            elif any(key in name for key in ['c_attn', 'c_proj', 'c_fc', 'lm_head']):
                # Scale learning rate by fan_in for weight matrices
                fan_in = param.size(1)
                group['lr'] = learning_rate / fan_in
            else:
                assert False, f"Parameter {name} not assigned to a group"
            
            peak_lrs[name] = group['lr']
            param_groups.append(group)
        
        # Log parameter counts and learning rates
        decay_params = [g for g in param_groups if g['weight_decay'] > 0]
        nodecay_params = [g for g in param_groups if g['weight_decay'] == 0]
        
        num_decay_params = sum(p.numel() for g in decay_params for p in g['params'])
        num_nodecay_params = sum(p.numel() for g in nodecay_params for p in g['params'])
        
        # Create AdamW optimizer with fused implementation if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=1e-20, **extra_args)
        print(f"\nusing fused AdamW: {use_fused}")
        
        return optimizer, peak_lrs

    @torch.no_grad()
    def generate(
        self,
        idx,
        tokens_to_generate,
        *,
        temperature=1.0,
        top_k=None,
        return_logits=False,
    ):
        """Autoregressively extend ``idx`` by ``tokens_to_generate`` tokens.

        Args:
            idx (torch.LongTensor): Conditioning tokens of shape (B, T_cond).
            tokens_to_generate (int): Number of tokens to sample.
            temperature (float, optional): Sampling temperature; 0 switches to greedy decoding.
            top_k (int, optional): If set, restrict sampling to the top-k logits before sampling.
            return_logits (bool, optional): When True, also return the raw logits used at each newly
                generated step. Shape will be (B, tokens_to_generate, V).
        """
        if tokens_to_generate < 0:
            raise ValueError("tokens_to_generate must be non-negative")
        if idx.dim() != 2:
            raise ValueError("idx must have shape (batch, sequence_length)")

        _, seq_len = idx.size()
        if seq_len == 0:
            raise ValueError("conditioning sequence must contain at least one token")
        if seq_len > self.config.S:
            raise ValueError(
                f"Conditioning sequence length {seq_len} exceeds block size {self.config.S}"
            )
        if seq_len + tokens_to_generate > self.config.S:
            raise ValueError(
                f"Requested {seq_len + tokens_to_generate} tokens exceeds block size {self.config.S}"
            )
        if tokens_to_generate == 0:
            return idx

        generated = idx.clone()

        logits, past_key_values = self._forward_with_past(
            generated,
            use_cache=True,
        )
        next_token_logits = logits[:, -1, :]

        collected_logits = [] if return_logits else None

        for step in range(tokens_to_generate):
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                logits_to_sample = next_token_logits / temperature
                if top_k is not None:
                    top_values, _ = torch.topk(
                        logits_to_sample, min(top_k, logits_to_sample.size(-1))
                    )
                    logits_to_sample = logits_to_sample.masked_fill(
                        logits_to_sample < top_values[:, [-1]],
                        float('-inf'),
                    )
                probs = F.softmax(logits_to_sample, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if return_logits:
                collected_logits.append(next_token_logits.clone())

            generated = torch.cat((generated, next_token), dim=1)

            if step + 1 == tokens_to_generate:
                break

            logits, past_key_values = self._forward_with_past(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = logits[:, -1, :]

        if return_logits:
            if collected_logits:
                logits_tensor = torch.stack(collected_logits, dim=1)
            else:
                logits_tensor = next_token_logits.new_empty((generated.size(0), 0, next_token_logits.size(-1)))
            return generated, logits_tensor

        return generated
