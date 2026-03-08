import hashlib
from plum import dispatch

@dispatch
def sha_hash(n: int):
    n_bytes = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    hash_bytes = hashlib.sha256(n_bytes).digest()
    hash_integer = int.from_bytes(hash_bytes, 'big')
    return int(hash_integer % (2**32 - 1))

@dispatch
def sha_hash(x: tuple):
    c = 1283989857
    for elem in x:
        c = c ^ sha_hash(elem)
    return c