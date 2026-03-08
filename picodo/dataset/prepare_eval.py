import os, re, io, pickle, shutil, urllib.request, numpy as np
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

# ─────────── config ───────────────────────────────────────────────────────────
out_dir       = "./fen2pv"
ctx_len       = 512                  # fixed sequence length (tokens)
train_size    = 100_000                 # use all – test_size examples
test_size     = 16_384
seed          = 42
num_proc      = 64                   # HF map / filter workers
num_threads   = 64                   # ThreadPoolExecutor for I/O
batch_size    = 1_024                # mem‑map write granularity

# ─────────── vocab & encoding ─────────────────────────────────────────────────
allowed_tokens = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')',
    'l', 'k', 'v', ':', '*', '|', 'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
BOS, EOS = '<BOS>', '<EOS>'
allowed_tokens = [BOS, EOS] + allowed_tokens
stoi = {tok: i for i, tok in enumerate(allowed_tokens)}
itos = {i: tok for tok, i in stoi.items()}
BOS_ID, EOS_ID = stoi[BOS], stoi[EOS]

_whitespace_re = re.compile(r'\s+')

def encode(text: str, *, add_bos=True, add_eos=True, pad_to=None):
    n_special = (1 if add_bos else 0) + (1 if add_eos else 0)
    exp_len = len(text) + n_special
    L = pad_to if pad_to is not None else exp_len
    arr = np.full(L, EOS_ID, np.uint8)
    idx = 0
    if add_bos:
        arr[idx] = BOS_ID; idx += 1
    for ch in text:
        if idx >= L: break
        arr[idx] = stoi.get(ch, stoi['?']); idx += 1
    if add_eos and idx < L:
        arr[idx] = EOS_ID; idx += 1
    return arr

def decode(arr: np.ndarray) -> str:
    return ''.join(itos[i] for i in arr)

# ─────────── helpers ──────────────────────────────────────────────────────────

def format_pv(line: str) -> str:
    """Convert space‑sep UCI PV to comma/semicolon alternation."""
    moves = line.split()
    res = []
    for i, mv in enumerate(moves):
        res.append(mv + (',' if i % 2 == 0 else ';'))
    return ''.join(res)

# ─────────── per‑example processing ───────────────────────────────────────────

def process(ex):
    fen = ex['fen']
    pv  = format_pv(ex['line'])
    text = f"{fen}|{pv}"
    tmp = encode(text, add_bos=True, add_eos=True)  # length check
    if len(tmp) > ctx_len:
        return {'ids': None, 'mask': None, 'len': len(tmp), 'skip': True}

    ids  = encode(text, add_bos=True, add_eos=True, pad_to=ctx_len)
    # mask: 1 for PV chars (after '|', before EOS)
    prefix = encode(f"{fen}|", add_bos=True, add_eos=False)
    start  = len(prefix)
    pv_len = len(encode(pv, add_bos=False, add_eos=False))
    end    = min(start + pv_len, ctx_len - 1)  # leave EOS & pad unmasked
    mask   = np.zeros(ctx_len, bool)
    mask[start:end] = True

    return {'ids': ids, 'mask': mask, 'len': len(tmp), 'skip': False}

# ─────────── mem‑map writer ───────────────────────────────────────────────────

def write_batch(batch, t_arr, m_arr, idx):
    for ex in batch:
        t_arr[idx:idx+ctx_len] = ex['ids']
        m_arr[idx:idx+ctx_len] = ex['mask']
        idx += ctx_len
    return idx

# ─────────── main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    print(f"Building FEN→PV dataset → {out_dir}")

    ds = load_dataset("Lichess/chess-position-evaluations", split=f"train[:{train_size}]" if train_size is not None else "train", num_proc=num_proc)
    print(f"Raw examples: {len(ds)/1e6:.2f}M")

    ds = ds.map(process, num_proc=num_proc, remove_columns=list(ds.features), desc="process")
    ds = ds.filter(lambda x: not x['skip'], num_proc=num_proc, desc="filter valid")
    print(f"Kept examples: {len(ds)/1e6:.2f}M")

    ds = ds.shuffle(seed=seed)
    if train_size is None:
        train_size = len(ds) - test_size
    splits = {
        'train': ds.select(range(train_size - test_size)),
        'test' : ds.select(range(train_size - test_size, train_size))
    }

    for split, d in splits.items():
        total = len(d) * ctx_len
        print(f"{split}: {total/1e9:.2f}B tokens, {len(d)/1e6:.2f}M ex")
        tok_path  = os.path.join(out_dir, f"{split}.bin")
        mask_path = os.path.join(out_dir, f"{split}_mask.bin")
        tok_mm  = np.memmap(tok_path,  dtype=np.uint8, mode='w+', shape=(total,))
        mask_mm = np.memmap(mask_path, dtype=bool,    mode='w+', shape=(total,))
        idx = 0
        for i in tqdm(range(0, len(d), batch_size), desc=f"write {split}"):
            batch = [d[j] for j in range(i, min(i+batch_size, len(d)))]
            idx = write_batch(batch, tok_mm, mask_mm, idx)
        tok_mm.flush(); mask_mm.flush()

    meta = {
        'vocab_size': len(allowed_tokens),
        'bos_token_id': BOS_ID,
        'eos_token_id': EOS_ID,
        'seed': seed,
        'train_size': train_size,
        'test_size': test_size,
        'ctx_len': ctx_len,
        'format': 'fen|pv_line'
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Done.")
