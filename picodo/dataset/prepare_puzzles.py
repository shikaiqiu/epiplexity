import os
import re
from tqdm import tqdm
import numpy as np
import pickle
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

out_dir = './puzzles'
os.makedirs(out_dir, exist_ok=True)
num_proc = 64
num_threads = 64 #min(32, os.cpu_count() + 4)
seed = 42
train_size = None
test_size = 16384
batch_size = 1024  # Increased batch size
ctx_len = 512  # Fixed context length for all examples

# Add FEN notation specific tokens
allowed_tokens = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')',
    'l', 'k', 'v', ':', '*', '|', 'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
allowed_tokens = [BOS_TOKEN, EOS_TOKEN] + allowed_tokens

# Create a mapping from tokens to integers - use array for faster lookup
stoi = {token: i for i, token in enumerate(allowed_tokens)}
itos = {i: token for i, token in enumerate(allowed_tokens)}

# Get the BOS and EOS token IDs
BOS_TOKEN_ID = stoi[BOS_TOKEN]
EOS_TOKEN_ID = stoi[EOS_TOKEN]

# Pre-compile regular expressions
move_number_pattern = re.compile(r'\d+\.\.?\.?\s*')
curly_braces_pattern = re.compile(r'\{[^}]*\}')
whitespace_pattern = re.compile(r'\s+')

# Create a lookup set for allowed tokens
allowed_tokens_set = set(allowed_tokens)

def encode(text, add_bos=True, add_eos=True, pad_to_length=None):
    """
    Encode text to token IDs and optionally pad to a fixed length with EOS tokens.
    """
    # Calculate expected length
    expected_length = len(text) + (1 if add_bos else 0) + (1 if add_eos else 0)
    
    # Set final length based on padding requirement
    final_length = max(expected_length, pad_to_length) if pad_to_length is not None else expected_length
    
    # Pre-allocate array of correct size
    result = np.zeros(final_length, dtype=np.uint8)
    idx = 0
    
    # Add BOS token if requested
    if add_bos:
        result[idx] = BOS_TOKEN_ID
        idx += 1
    
    # Add the text tokens
    for char in text:
        result[idx] = stoi[char]
        idx += 1
    
    # Add EOS token if requested
    if add_eos:
        result[idx] = EOS_TOKEN_ID
        idx += 1
    
    # Fill the rest with EOS tokens for padding
    while idx < final_length:
        result[idx] = EOS_TOKEN_ID
        idx += 1
    
    return result

def decode(ids):
    """Decode token IDs back to text."""
    return ''.join([itos[i] for i in ids])

def fast_format_moves(moves):
    """Format moves with a single pass through the array"""
    if not moves:
        return ""
    
    # Pre-allocate result with estimated size (average move length ~3 chars + separator)
    result = []
    for i, move in enumerate(moves):
        # Append move with appropriate separator
        if i % 2 == 0:  # White's move
            result.append(move + ",")
        else:  # Black's move
            result.append(move + ";")
    
    return ''.join(result)

def process(example):
    # Extract what we need
    original_movetext = example['ctx'] + example['target']
    
    # Clean text with single regex pass
    cleaned_text = curly_braces_pattern.sub('', move_number_pattern.sub('', original_movetext))
    cleaned_text = whitespace_pattern.sub(' ', cleaned_text).strip()
    
    # Get moves and format in one step
    moves = cleaned_text.split()
    formatted_text = fast_format_moves(moves)[:-1] # remove trailing separator
    
    # Encode efficiently and pad to ctx_len
    ids = encode(formatted_text, add_bos=True, add_eos=True, pad_to_length=ctx_len)

    # build a boolean mask only for target moves
    seq_len = len(ids)
    mask = np.zeros(seq_len, dtype=np.bool_)
    # the last separator index by finding the last ; or , which ever comes last
    target_start_idx = max(formatted_text.rfind(';'), formatted_text.rfind(',')) + 2 # add one for bos and one to move past the separator
    # find first eos token 
    target_end_idx = np.where(ids == EOS_TOKEN_ID)[0][0]
    mask[target_start_idx:target_end_idx] = True

    return {
        'ids': ids,
        'mask': mask,
        'len': seq_len  # Always return fixed length since we're padding
    }

def process_and_write_batch(batch, tokens_arr, mask_arr, start_idx):
    """Process a batch of examples and write results at once"""
    end_idx = start_idx
    all_tokens = []
    all_masks = []
    total_len = 0
    
    # Process all examples in the batch
    for example in batch:
        ids = np.array(example['ids'], dtype=np.uint8)
        all_tokens.append(ids)
        all_masks.append(example['mask'])
        total_len += len(ids)
    
    # Flatten and write
    flat_tokens = np.concatenate(all_tokens, dtype=np.uint8)
    flat_masks = np.concatenate(all_masks, dtype=np.bool_)
    end_idx = start_idx + total_len
    tokens_arr[start_idx:end_idx] = flat_tokens
    mask_arr[start_idx:end_idx] = flat_masks
    return end_idx

if __name__ == '__main__':
    print(f"Using {num_proc} processes for dataset operations and {num_threads} threads for I/O")
    print(f"Padding all examples to fixed context length of {ctx_len}")
    
    # Load the dataset - use streaming for even less memory usage if possible
    dataset = load_dataset("EleutherAI/lichess-puzzles", split='train', num_proc=num_proc)
    
    print(f"Total examples: {len(dataset)/1e6}M")
    
    # Process dataset with optimized map function
    processed_dataset = dataset.map(
        process,
        num_proc=num_proc,
        desc="processing dataset",
        batch_size=1000  # Use batching within map
    )
    # remove examples that are too long
    processed_dataset = processed_dataset.filter(lambda x: x['len'] <= ctx_len)
    
    # Split into train and test
    processed_dataset = processed_dataset.shuffle(seed=seed)
    if train_size is None:
        train_size = len(processed_dataset) - test_size
    splits = {
        'train': processed_dataset.select(range(train_size - test_size)),
        'test': processed_dataset.select(range(train_size - test_size, train_size))
    }
    
    # Save each split with highly optimized batch writing
    for split, ds in splits.items():
        tokens_filename = os.path.join(out_dir, f'{split}.bin')
        mask_filename = os.path.join(out_dir, f'{split}_mask.bin')
        # Calculate total length of all sequences (should be fixed length * num examples)
        total_tokens = len(ds) * ctx_len
        print(f'{split}: {total_tokens/1e9}B tokens, {len(ds)/1e6}M examples')
        
        # Create memory map for the tokens
        tokens_dtype = np.uint8
        mask_dtype = np.bool_
        tokens_arr = np.memmap(tokens_filename, dtype=tokens_dtype, mode='w+', shape=(total_tokens,))
        mask_arr = np.memmap(mask_filename, dtype=mask_dtype, mode='w+', shape=(total_tokens,))
        
        # Use larger batch size and process with threads
        token_idx = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process in larger chunks for better throughput
            for i in tqdm(range(0, len(ds), batch_size), desc=f'writing {split} data'):
                batch = ds.select(range(i, min(i + batch_size, len(ds))))
                end_idx = process_and_write_batch(batch, tokens_arr, mask_arr, token_idx)
                token_idx = end_idx
        
        # Ensure data is written
        tokens_arr.flush()
        mask_arr.flush()
    # Save metadata
    meta = {
        'vocab_size': len(allowed_tokens),
        'bos_token_id': BOS_TOKEN_ID,
        'eos_token_id': EOS_TOKEN_ID,
        'seed': seed,
        'train_size': train_size,
        'test_size': test_size,
        'ctx_len': ctx_len,  # Add context length to metadata
        'format': 'moves'
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Dataset preparation complete!")