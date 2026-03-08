import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from omegaconf.dictconfig import DictConfig
from typing import List, Optional, Dict, Any, Tuple


class TransformerDecoder(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    self.blocks = [TransformerBlock(cfg, rngs) for _ in range(cfg.N)]
    self.out_ln = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.readout = nnx.Linear(in_features=cfg.D, out_features=cfg.V, use_bias=False, kernel_init=fsdp_init('readout', cfg), dtype=cfg.dtype, rngs=rngs)

  def get_features(self, x):
    h = self.embed(x) + self.pos_embed(jnp.arange(x.shape[1])[None, ...])
    for block in self.blocks:
      h = block(h)
    return h

  def __call__(self, x):
    h = self.get_features(x)
    h = self.out_ln(h)
    return self.readout(h)

  def init_cache(self, batch_size: int, max_len: int) -> Tuple[Dict[str, Any], ...]:
    dtype = self.embed.embedding.value.dtype
    caches = [block.init_cache(batch_size, max_len, dtype) for block in self.blocks]
    return tuple(caches)

  def forward_step(
      self,
      token_ids: jax.Array,
      position: jax.Array,
      caches: List[Dict[str, Any]],
  ) -> Tuple[jax.Array, List[Dict[str, Any]]]:
    token_ids = token_ids.astype(jnp.int32)
    token_inputs = token_ids[:, None]
    x = self.embed(token_inputs)[:, 0, :]
    position_ids = jnp.full((token_ids.shape[0], 1), position, dtype=jnp.int32)
    pos_vec = self.pos_embed(position_ids)[:, 0, :]
    x = x + pos_vec
    new_caches: List[Dict[str, Any]] = []
    for block, cache in zip(self.blocks, caches):
      x, updated_cache = block.forward_step(x, cache, position)
      new_caches.append(updated_cache)
    x = self.out_ln(x)
    logits = self.readout(x)
    return logits, tuple(new_caches)


class Attention(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.num_heads = cfg.D // cfg.dh
    self.head_dim = cfg.dh
    self.scale = (1 / self.head_dim) ** 0.5
    self.query_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.key_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.value_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.output_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('readout', cfg), dtype=cfg.dtype, rngs=rngs)
    self.q_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.k_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)

  def __call__(self, x, mask=None):
    B, S, D = x.shape
    H = self.num_heads

    q = self.query_proj(x)
    k = self.key_proj(x)
    v = self.value_proj(x)

    q = q.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, H, -1).transpose(0, 2, 1, 3)

    q = self.q_norm(q)
    k = self.k_norm(k)

    attn_scores = jnp.einsum('bhsd,bhtd->bhst', q, k) * self.scale
    if mask is not None: attn_scores = jnp.where(mask, attn_scores, jnp.finfo(attn_scores.dtype).min)

    attn_probs = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum('bhst,bhtd->bhsd', attn_probs, v)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    return self.output_proj(out)

  def init_cache(self, batch_size: int, max_len: int, dtype: jnp.dtype) -> Dict[str, Any]:
    k = jnp.zeros((batch_size, self.num_heads, max_len, self.head_dim), dtype=dtype)
    v = jnp.zeros((batch_size, self.num_heads, max_len, self.head_dim), dtype=dtype)
    return {'k': k, 'v': v}

  def forward_step(
      self,
      x: jax.Array,
      cache: Dict[str, Any],
      position: jax.Array,
  ) -> Tuple[jax.Array, Dict[str, Any]]:
    position = jnp.asarray(position, dtype=jnp.int32)
    B = x.shape[0]
    q = self.query_proj(x)
    k_new = self.key_proj(x)
    v_new = self.value_proj(x)

    q = q.reshape(B, self.num_heads, self.head_dim)
    k_new = k_new.reshape(B, self.num_heads, self.head_dim)
    v_new = v_new.reshape(B, self.num_heads, self.head_dim)

    q = self.q_norm(q)
    k_new = self.k_norm(k_new)

    k_cache = cache['k'].at[:, :, position, :].set(k_new)
    v_cache = cache['v'].at[:, :, position, :].set(v_new)

    attn_scores = jnp.einsum('bhd,bhTd->bhT', q, k_cache) * self.scale
    valid_mask = (jnp.arange(k_cache.shape[2]) <= position).astype(attn_scores.dtype)
    large_neg = jnp.finfo(attn_scores.dtype).min
    attn_scores = jnp.where(valid_mask[None, None, :], attn_scores, large_neg)
    attn_probs = jax.nn.softmax(attn_scores, axis=-1)
    context = jnp.einsum('bhT,bhTd->bhd', attn_probs, v_cache)
    context = context.reshape(B, -1)
    out = self.output_proj(context)

    return out, {'k': k_cache, 'v': v_cache}


class TransformerBlock(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.ln1 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.attn = Attention(cfg, rngs)
    self.ln2 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.mlp = Mlp(cfg, rngs)
    self.branch_multiplier = 1 / cfg.N

  def __call__(self, x):
    h = self.ln1(x)
    mask = nnx.make_causal_mask(jnp.ones((x.shape[0], x.shape[1])), dtype=x.dtype)
    x = x + self.attn(h, mask=mask) * self.branch_multiplier
    return x + self.mlp(self.ln2(x)) * self.branch_multiplier

  def init_cache(self, batch_size: int, max_len: int, dtype: jnp.dtype) -> dict:
    return {'attn': self.attn.init_cache(batch_size, max_len, dtype)}

  def forward_step(
      self,
      x: jax.Array,
      cache: dict,
      position: jax.Array,
  ) -> Tuple[jax.Array, Dict[str, Any]]:
    h = self.ln1(x)
    attn_out, attn_cache = self.attn.forward_step(h, cache['attn'], position)
    x = x + attn_out * self.branch_multiplier
    x = x + self.mlp(self.ln2(x)) * self.branch_multiplier
    return x, {'attn': attn_cache}


class Mlp(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(in_features=cfg.D, out_features=4*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_kernel', cfg), dtype=cfg.dtype, rngs=rngs)
    self.fc2 = nnx.Linear(in_features=4*cfg.D, out_features=cfg.D, use_bias=False, kernel_init=fsdp_init('readout', cfg), dtype=cfg.dtype, rngs=rngs)

  def __call__(self, x):
    h = jax.nn.gelu(self.fc1(x))
    return self.fc2(h)


def fsdp_init(layer_type: str, cfg: DictConfig):
  partition_fn = nnx.with_partitioning if cfg.fsdp_enabled else lambda x, _: x
  kernel_init = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="normal")
  embed_init = jax.nn.initializers.normal(stddev=cfg.embed_init_std)
  readout_init = jax.nn.initializers.zeros
  match layer_type:
    case "embedding":
      return partition_fn(embed_init, (None, "data"))
    case "attn_proj":
      return partition_fn(kernel_init, ("data", None))
    case "mlp_kernel":
      return partition_fn(kernel_init, ("data", None))
    case "readout":
      return partition_fn(readout_init, ("data", None))
    case _:
      raise ValueError(f"unrecognized layer type: {layer_type}")


def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
  @nnx.jit
  def initialize_sharded_model():
    model = TransformerDecoder(c, rngs=nnx.Rngs(seed))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

  with mesh:
    model = initialize_sharded_model()

  return model


def generate(model: TransformerDecoder,
             batch_size: int,
             seq_len: int,
             rng: jax.Array,
             bos_token: int = 0,
             temperature: float = 1.0,
             data_sharding=None,
             use_kv_cache: bool = True):
  if seq_len <= 0:
    raise ValueError("seq_len must be positive")

  def _generate_without_cache(tokens: jax.Array, rng_in: jax.Array):
    def body(carry, idx):
      toks, rng_curr = carry
      logits_full = model(toks)
      step_logits = logits_full[:, idx, :]
      if temperature != 1.0:
        step_logits = step_logits / temperature
      rng_next, step_rng = jax.random.split(rng_curr)
      sampled = jax.random.categorical(step_rng, step_logits).astype(toks.dtype)
      toks = toks.at[:, idx + 1].set(sampled)
      return (toks, rng_next), None

    if seq_len > 1:
      (tokens_out, rng_out), _ = jax.lax.scan(
          body,
          (tokens, rng_in),
          jnp.arange(seq_len - 1, dtype=jnp.int32),
      )
    else:
      tokens_out, rng_out = tokens, rng_in
    teacher_logits = model(tokens_out)
    return tokens_out, teacher_logits, rng_out

  tokens = jnp.full((batch_size, seq_len), bos_token, dtype=jnp.uint16)
  if data_sharding is not None:
    tokens = jax.device_put(tokens, data_sharding)
  tokens = tokens.at[:, 0].set(bos_token)

  if not use_kv_cache:
    return _generate_without_cache(tokens, rng)

  caches = model.init_cache(batch_size, seq_len)
  if data_sharding is not None:
    caches = jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), caches)

  seq_indices = jnp.arange(seq_len, dtype=jnp.int32)

  def scan_step(carry, idx):
    tokens_carry, caches_carry, rng_carry = carry
    token_ids = tokens_carry[:, idx]
    logits_step, caches_next = model.forward_step(token_ids, idx, caches_carry)

    logits_for_sampling = logits_step / temperature if temperature != 1.0 else logits_step
    rng_next, step_rng = jax.random.split(rng_carry)
    sampled = jax.random.categorical(step_rng, logits_for_sampling).astype(tokens_carry.dtype)

    def update_tokens(toks):
      return toks.at[:, idx + 1].set(sampled)

    tokens_next = jax.lax.cond(
        idx < (seq_len - 1),
        update_tokens,
        lambda toks: toks,
        tokens_carry,
    )

    new_carry = (tokens_next, caches_next, rng_next)
    return new_carry, logits_step

  (tokens, caches, rng), logits_history = jax.lax.scan(
      scan_step,
      (tokens, caches, rng),
      seq_indices,
  )

  teacher_logits = jnp.swapaxes(logits_history, 0, 1)
  return tokens, teacher_logits, rng
