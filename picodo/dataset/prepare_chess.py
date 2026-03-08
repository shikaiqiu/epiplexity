import os
import re
from tqdm import tqdm
import numpy as np
import pickle
from datasets import load_dataset
import chess
import chess.pgn
import io
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

out_dir = './chess'
os.makedirs(out_dir, exist_ok=True)
num_proc = 64
num_threads = min(32, os.cpu_count() + 4)
seed = 42
train_size = None
test_size = 16384
batch_size = 10000

allowed_tokens = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')',
    'l', 'k', 'v', ':', '*', '|', 'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
allowed_tokens = [BOS_TOKEN, EOS_TOKEN] + allowed_tokens

stoi = {token: i for i, token in enumerate(allowed_tokens)}
itos = {i: token for i, token in enumerate(allowed_tokens)}

BOS_TOKEN_ID = stoi[BOS_TOKEN]
EOS_TOKEN_ID = stoi[EOS_TOKEN]

move_number_pattern = re.compile(r'\d+\.\.?\.?\s*')
curly_braces_pattern = re.compile(r'\{[^}]*\}')
whitespace_pattern = re.compile(r'\s+')

allowed_tokens_set = set(allowed_tokens)

def encode(text, add_bos=True, add_eos=True):
    result = np.zeros(len(text) + (1 if add_bos else 0) + (1 if add_eos else 0), dtype=np.uint8)
    idx = 0
    if add_bos:
        result[idx] = BOS_TOKEN_ID
        idx += 1
    for char in text:
        result[idx] = stoi[char]
        idx += 1
    if add_eos:
        result[idx] = EOS_TOKEN_ID
        idx += 1
    return result[:idx]

@lru_cache(maxsize=100000)
def get_final_position(raw_movetext):
    try:
        pgn_text = "[Event \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n" + raw_movetext
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
        board = game.end().board()
        return board.fen()
    except Exception:
        return None

def fast_format_moves(moves):
    if not moves:
        return ""
    result = []
    for i, move in enumerate(moves):
        if i % 2 == 0:
            result.append(move + ",")
        else:
            result.append(move + ";")
    return ''.join(result)

def process(example):
    original_movetext = example['movetext']
    cleaned_text = curly_braces_pattern.sub('', move_number_pattern.sub('', original_movetext))
    cleaned_text = whitespace_pattern.sub(' ', cleaned_text).strip()
    moves = cleaned_text.split()
    formatted_text = fast_format_moves(moves)
    final_position = get_final_position(original_movetext)
    white_elo = str(example['WhiteElo'])
    black_elo = str(example['BlackElo'])
    if final_position:
        combined_text = formatted_text + "|" + white_elo + "|" + black_elo + "|" + final_position
    else:
        combined_text = formatted_text
    ids = encode(combined_text, add_bos=True, add_eos=True)
    return {
        'ids': ids,
        'len': len(ids)
    }

def process_and_write_batch(batch, tokens_arr, start_idx):
    end_idx = start_idx
    all_tokens = []
    total_len = 0
    for example in batch:
        ids = np.array(example['ids'], dtype=np.uint8)
        all_tokens.append(ids)
        total_len += len(ids)
    flat_tokens = np.concatenate(all_tokens, dtype=np.uint8)
    end_idx = start_idx + total_len
    tokens_arr[start_idx:end_idx] = flat_tokens
    return end_idx

if __name__ == '__main__':
    print(f"Using {num_proc} processes for dataset operations and {num_threads} threads for I/O")

    base_url = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year=2025/month=01/"
    data_files = {"train": [base_url + f"train-{i:05d}-of-00072.parquet" for i in range(36, 72)]}

    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        split=f"train[:{train_size}]" if train_size is not None else "train",
        num_proc=num_proc
    )

    print(f"Total examples: {len(dataset)/1e6}M")

    processed_dataset = dataset.map(
        process,
        remove_columns=['movetext'],
        num_proc=num_proc,
        desc="processing dataset",
        batch_size=1000
    )

    processed_dataset = processed_dataset.shuffle(seed=seed)
    if train_size is None:
        train_size = len(processed_dataset) - test_size

    splits = {
        'train': processed_dataset.select(range(train_size - test_size)),
        'test': processed_dataset.select(range(train_size - test_size, train_size))
    }

    for split, ds in splits.items():
        tokens_filename = os.path.join(out_dir, f'{split}.bin')
        arr_len = np.sum(ds['len'], dtype=np.uint64)
        print(f'{split}: {arr_len/1e9}B tokens, {len(ds)/1e6}M examples')
        tokens_dtype = np.uint8
        tokens_arr = np.memmap(tokens_filename, dtype=tokens_dtype, mode='w+', shape=(arr_len,))

        token_idx = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in tqdm(range(0, len(ds), batch_size), desc=f'writing {split} data'):
                batch = ds.select(range(i, min(i + batch_size, len(ds))))
                end_idx = process_and_write_batch(batch, tokens_arr, token_idx)
                token_idx = end_idx

        tokens_arr.flush()

    meta = {
        'vocab_size': len(allowed_tokens),
        'bos_token_id': BOS_TOKEN_ID,
        'eos_token_id': EOS_TOKEN_ID,
        'seed': seed,
        'train_size': train_size,
        'test_size': test_size,
        'format': 'moves|white_elo|black_elo|fen_position'
    }

    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset preparation complete!")
