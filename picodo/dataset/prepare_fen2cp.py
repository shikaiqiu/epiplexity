import os, pickle, numpy as np, re
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ─────────── config ───────────────────────────────────────────────────────────
out_dir       = "./fen2cp"
ctx_len       = 128           # more than enough for FEN + single‑digit label
train_size    = 100_000          # use all minus test_size
test_size     = 16_384
seed          = 42
num_proc      = 64
batch_size    = 1_024         # memmap write chunk

# ─────────── vocabulary (same as other datasets) ──────────────────────────────
allowed_tokens = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')',
    'l', 'k', 'v', ':', '*', '|', 'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
BOS, EOS = '<BOS>', '<EOS>'
allowed_tokens = [BOS, EOS] + allowed_tokens
stoi = {t: i for i, t in enumerate(allowed_tokens)}
itos = {i: t for t, i in stoi.items()}
BOS_ID, EOS_ID = stoi[BOS], stoi[EOS]

# ─────────── encoding helpers ─────────────────────────────────────────────────

def encode(text: str, *, add_bos=True, add_eos=True, pad_to=None):
    special = (1 if add_bos else 0) + (1 if add_eos else 0)
    exp_len = len(text) + special
    L = pad_to if pad_to is not None else exp_len
    arr = np.full(L, EOS_ID, np.uint8)
    i = 0
    if add_bos:
        arr[i] = BOS_ID; i += 1
    for ch in text:
        if i >= L: break
        arr[i] = stoi.get(ch, stoi['?']); i += 1
    if add_eos and i < L:
        arr[i] = EOS_ID
    return arr

# ─────────── class mapping ────────────────────────────────────────────────────
# Piece‑based symmetric buckets (option 1)
# 0: cp <= −800   …  4: −50..+50  …  8: cp >= +800
bounds = [-800, -400, -200, -50, 50, 200, 400, 800]

def cp_to_class(cp: int | None, mate: int | None) -> int:
    """Map Stockfish evaluation to class id 0‑8. Mate counts as extreme bucket."""
    # mate wins/loses override cp
    if mate is not None:
        if mate > 0:  # side to move mates
            return 8
        elif mate < 0:  # getting mated
            return 0
    # Normal centipawn path
    if cp is None:
        return 4  # treat missing as equal
    if cp <= bounds[0]:
        return 0
    if cp >= bounds[-1]:
        return 8
    # inside range
    for idx in range(1, len(bounds)):
        if bounds[idx-1] < cp <= bounds[idx]:
            return idx
    # (+50..+200 etc handled)
    return 4  # fallback equal

# ─────────── per‑example processing ───────────────────────────────────────────

def process(ex):
    cls = cp_to_class(ex.get('cp'), ex.get('mate'))
    text = f"{ex['fen']}|{cls}"
    raw = encode(text, add_bos=True, add_eos=True)
    if len(raw) > ctx_len:
        return {"skip": True}
    ids  = encode(text, add_bos=True, add_eos=True, pad_to=ctx_len)
    # mask only the class digit(s)
    prefix_len = len(encode(f"{ex['fen']}|", add_bos=True, add_eos=False))
    class_len  = len(encode(str(cls), add_bos=False, add_eos=False))
    mask = np.zeros(ctx_len, bool)
    mask[prefix_len:prefix_len+class_len] = True
    return {"ids": ids, "mask": mask, "skip": False}

# ─────────── mem‑map writing ──────────────────────────────────────────────────

def write_batch(batch, tok_mm, mask_mm, idx):
    for ex in batch:
        tok_mm[idx:idx+ctx_len] = ex['ids']
        mask_mm[idx:idx+ctx_len] = ex['mask']
        idx += ctx_len
    return idx

# ─────────── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    print("Building FEN→cp‑class dataset…")

    ds = load_dataset("Lichess/chess-position-evaluations", split=f"train[:{train_size}]" if train_size is not None else "train", num_proc=num_proc)
    ds = ds.map(process, num_proc=num_proc, remove_columns=list(ds.features), desc="process")
    ds = ds.filter(lambda x: not x['skip'], num_proc=num_proc, desc="filter")
    ds = ds.shuffle(seed=seed)

    if train_size is None:
        train_size = len(ds) - test_size
    splits = {
        'train': ds.select(range(train_size - test_size)),
        'test': ds.select(range(train_size - test_size, train_size))
    }

    for name, d in splits.items():
        N = len(d)*ctx_len
        print(f"{name}: {N/1e9:.2f}B tokens, {len(d)/1e6:.2f}M ex")
        tok_path  = os.path.join(out_dir, f"{name}.bin")
        mask_path = os.path.join(out_dir, f"{name}_mask.bin")
        tok_mm  = np.memmap(tok_path,  dtype=np.uint8, mode='w+', shape=(N,))
        mask_mm = np.memmap(mask_path, dtype=bool,    mode='w+', shape=(N,))
        idx=0
        for i in tqdm(range(0,len(d),batch_size), desc=f"write {name}"):
            batch=[d[j] for j in range(i, min(i+batch_size,len(d)))]
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
        'format': 'fen|cp_class (0‑8)'
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Done.")
