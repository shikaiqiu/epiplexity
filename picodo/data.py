import jax
import numpy as np
import jax.numpy as jnp
import os

def make_dummy_ds_loader(seq_len, batch_size):
    """Creates a dummy data loader that returns batches of zeros."""
    
    # Set a reasonable number of tokens for the dummy dataset
    n_tokens = batch_size * seq_len * 100000  # Arbitrary large number
    
    def get_batch(idx):
        # Return a batch of zeros with shape [batch_size, seq_len]
        return np.ones((batch_size, seq_len), dtype=np.uint16)
    
    return get_batch, n_tokens

def make_ds_loader(ds_path, split, seq_len, batch_size, bos_id=0):
    """note: we assume that the dataset on the disk is already shuffled!"""

    # get num. tokens
    data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')
    n_tokens = len(data)

    def get_batch(idx):
        # read dataset
        # using np.memmap for each batch to avoid memory leak
        data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')

        # get batch
        max_idx = n_tokens // (batch_size * seq_len)
        idx = idx % max_idx  # wrap around if idx is out of bounds
        
        start_idx = batch_size*seq_len*idx + seq_len*np.arange(batch_size)
        token_idx = start_idx[:, None] + np.arange(seq_len)[None, :] # [batch, sequence]
        tokens = data[token_idx]

        # add bos_id to the beginning of the batch
        bos = jnp.full((tokens.shape[0], 1), bos_id, dtype=tokens.dtype)
        tokens = jnp.concatenate([bos, tokens[:, :-1]], axis=1)
        mask = jnp.ones_like(tokens, dtype=jnp.bool_)

        return tokens, mask

    return get_batch, n_tokens

def make_downstream_ds_loader(ds_path, split, seq_len, batch_size):
    """note: we assume that the dataset on the disk is already shuffled!"""

    # get num. tokens
    data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')
    n_tokens = len(data)
    if os.path.exists(f'{ds_path}/{split}_mask.bin'):
        mask = np.memmap(f'{ds_path}/{split}_mask.bin', dtype=np.bool_, mode='r')
        print(f'Found mask for {ds_path}/{split}')
    else:
        mask = np.ones_like(data, dtype=np.bool_)
        print(f'No mask found for {ds_path}/{split}, using ones')

    def get_batch(idx):
        # read dataset
        # using np.memmap for each batch to avoid memory leak
        data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')
        mask = np.memmap(f'{ds_path}/{split}_mask.bin', dtype=np.bool_, mode='r')

        # get batch
        max_idx = n_tokens // (batch_size * seq_len)
        idx = idx % max_idx  # wrap around if idx is out of bounds
        
        start_idx = batch_size*seq_len*idx + seq_len*np.arange(batch_size)
        token_idx = start_idx[:, None] + np.arange(seq_len)[None, :] # [batch, sequence]
        tokens = data[token_idx]
        masks = mask[token_idx]

        return tokens, masks

    return get_batch, n_tokens

def get_in_out(batch: jax.Array, pad_id: int = 0, bos_id: int = 0):
    """Returns input, output, and weights for a batch of examples."""
    # Assumes input of the form <BOS> <IDs> <EOS> for eval.
    # in our datasets, <BOS> is 0, same as pad_id.
    # by masking out loss on pad_id, we also mask out loss on <BOS> as wanted.
    # x = batch # [B, L]
    if isinstance(batch, tuple):
        x, mask = batch
    else:
        x, mask = batch, None
    y = jnp.pad(x[:, 1:], ((0, 0), (0, 1)), constant_values=pad_id) # shift x by 1 along L axis
    if mask is not None:
        # shift mask by 1 along L axis
        weights = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False).astype(jnp.float32)
    else:
        # do train on <bos>
        # weights = jnp.ones_like(y, dtype=jnp.float32)
        weights = jnp.where(y != pad_id, 1, 0).astype(jnp.float32)
    return x, y, weights