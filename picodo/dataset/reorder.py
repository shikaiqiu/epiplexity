"""
Re-order records: moves|wElo|bElo|fen -> fen|moves|wElo|bElo
"""

import os, pickle
import numpy as np
from tqdm import tqdm

IN_DIR  = "./chess"
OUT_DIR = "./chess_reordered"
os.makedirs(OUT_DIR, exist_ok=True)

tok = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')', 'l', 'k', 'v', ':', '*', '|',
    'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
tok = ["<BOS>", "<EOS>"] + tok
itos = {i: t for i, t in enumerate(tok)}
stoi = {t: i for i, t in enumerate(tok)}
BOS_ID, EOS_ID, PIPE_ID = stoi["<BOS>"], stoi["<EOS>"], stoi["|"]


def reorder_token_record(seq: np.ndarray) -> np.ndarray:
    """Reorder: BOS moves|wElo|bElo|fen EOS -> BOS fen|moves|wElo|bElo EOS"""
    pipes = np.flatnonzero(seq == PIPE_ID)
    if pipes.size != 3:
        return seq.copy()

    moves = seq[1           : pipes[0]]
    wElo  = seq[pipes[0]+1  : pipes[1]]
    bElo  = seq[pipes[1]+1  : pipes[2]]
    fen   = seq[pipes[2]+1  : -1]

    out = np.empty_like(seq)
    pos = 0
    out[pos] = BOS_ID; pos += 1
    out[pos : pos+fen.size]   = fen;   pos += fen.size
    out[pos] = PIPE_ID;                pos += 1
    out[pos : pos+moves.size] = moves; pos += moves.size
    out[pos] = PIPE_ID;                pos += 1
    out[pos : pos+wElo.size]  = wElo;  pos += wElo.size
    out[pos] = PIPE_ID;                pos += 1
    out[pos : pos+bElo.size]  = bElo;  pos += bElo.size
    out[pos] = EOS_ID

    return out


with open(os.path.join(IN_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

for split in ("train", "test"):
    print(f"Re-ordering {split}")
    src_path = os.path.join(IN_DIR,  f"{split}.bin")
    dst_path = os.path.join(OUT_DIR, f"{split}.bin")

    src = np.memmap(src_path, dtype=np.uint8, mode="r")
    dst = np.memmap(dst_path, dtype=np.uint8, mode="w+", shape=src.shape)

    eos_pos = np.where(src == EOS_ID)[0]

    read_ptr  = 0
    write_ptr = 0
    for eos in tqdm(eos_pos, desc=split, unit="rec"):
        rec = src[read_ptr : eos+1]
        reordered = reorder_token_record(rec)
        dst[write_ptr : write_ptr + reordered.size] = reordered
        write_ptr += reordered.size
        read_ptr   = eos + 1

    dst.flush()
    print(f"  {split}: {write_ptr:,} tokens written")

meta["format"] = "fen|moves|white_elo|black_elo"
with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Done - reordered files are in", OUT_DIR)
