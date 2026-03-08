# save_cifar5m_memmap_no_bos.py
import os, random, numpy as np
from tqdm import tqdm


shard_pattern = '/data/users/shikai_q/datasets/cifar5m_part{}.npz' # REPLACE WITH YOUR PATH
num_shards    = 6
seed          = 2357
out_dir       = 'cifar5m'
train_path    = os.path.join(out_dir, 'train.bin')
test_path     = os.path.join(out_dir, 'test.bin')
dtype         = np.uint8


def to_tokens(rgb_batch: np.ndarray) -> np.ndarray:
    """RGB uint8 → flattened grayscale uint8 tokens (no BOS/EOS).
       In:  (B, 32, 32, 3) ──>  Out: (B, 1024)
    """
    gray = rgb_batch.mean(axis=-1).astype(np.uint8)    # simple avg (R+G+B)//3
    return gray.reshape(gray.shape[0], -1)


tot_imgs = 0
for i in range(num_shards):
    with np.load(shard_pattern.format(i), mmap_mode='r') as data:
        tot_imgs += data['X'].shape[0]

random.seed(seed)
test_idxs = set(random.sample(range(tot_imgs), k=2_000))

tokens_per_img = 32 * 32                             # 1024
train_imgs = tot_imgs - 2_000
train_tokens = train_imgs * tokens_per_img
test_tokens  = 2_000     * tokens_per_img

train_mm = np.memmap(train_path, dtype=dtype, mode='w+', shape=(train_tokens,))
test_mm  = np.memmap(test_path,  dtype=dtype, mode='w+', shape=(test_tokens,))


tri = tei = 0        # cursors in train.bin / test.bin
g_idx = 0            # global image index

for shard in range(num_shards):
    with np.load(shard_pattern.format(shard), mmap_mode='r') as data:
        X = data['X']                                   # (N,32,32,3)
        for start in tqdm(range(0, X.shape[0], 10_000),
                          desc=f'shard {shard}', unit='img'):
            batch = X[start:start + 10_000]
            flat  = to_tokens(batch).reshape(-1)         # 1-D token stream

            bsize = batch.shape[0]
            sel_test = [g_idx + k in test_idxs for k in range(bsize)]

            # write test images
            if any(sel_test):
                test_sel = flat[np.repeat(sel_test, tokens_per_img)]
                n = test_sel.size
                test_mm[tei:tei+n] = test_sel
                tei += n

            # write train images
            if not all(sel_test):
                train_sel = flat[np.repeat([not t for t in sel_test],
                                            tokens_per_img)]
                n = train_sel.size
                train_mm[tri:tri+n] = train_sel
                tri += n

            g_idx += bsize

train_mm.flush()
test_mm.flush()

print(f'done: {train_path} ({train_tokens:,} tokens)  |  {test_path} ({test_tokens:,} tokens)')