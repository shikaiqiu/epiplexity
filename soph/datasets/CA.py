# Cellular automaton data generation for experiments

import numpy as np


def int2bits(x, bits=8):
    """Convert integer to bit array (LSB first)."""
    bits_str = bin(x)[2:].zfill(bits)
    return np.array(list(map(int, bits_str[::-1])), dtype=np.uint8)


def evolve_CA(initial_state, steps, rule=30):
    """Evolve Rule k cellular automaton.

    Parameters:
    - initial_state: (..., N) array representing the initial state
    - steps: int, number of time steps to simulate.

    Returns: (..., N) numpy array representing the final state
    """
    lookup = int2bits(rule, 8)
    state = initial_state
    for _ in range(steps):
        left = np.roll(state, 1, axis=-1)
        right = np.roll(state, -1, axis=-1)
        neighborhood = (left << 2) + (state << 1) + right
        state = lookup[neighborhood]
    return state


def generate(B, steps, width, rule, burnin, sep, seed):
    """Generate batch of CA sequences: initial state -> final state after `steps`.

    Returns (B, 2*width + sep) array concatenating input and output.
    """
    rng = np.random.default_rng(seed)
    initial_states = rng.integers(0, 2, size=(B, width), dtype=np.uint8)
    if burnin:
        initial_states = evolve_CA(initial_states, 1000, rule)
    out_states = evolve_CA(initial_states, steps, rule)
    sep_tokens = 2 * np.ones((B, int(sep)), dtype=np.uint8)
    return np.concatenate((initial_states, sep_tokens, out_states), axis=-1)


def generate_freq_steps(B, steps, width, rule, burnin, sep, freq, seed):
    """Generate initial state and every `freq`-th intermediate state (always includes final)."""
    rng = np.random.default_rng(seed)
    initial_states = rng.integers(0, 2, size=(B, width), dtype=np.uint8)
    if burnin:
        initial_states = evolve_CA(initial_states, 1000, rule)

    target_steps = list(range(freq, steps + 1, freq))
    if target_steps[-1] != steps:
        target_steps.append(steps)

    current = initial_states
    collected = []
    last_step = 0
    for ts in target_steps:
        delta = ts - last_step
        current = evolve_CA(current, delta, rule)
        collected.append(current)
        last_step = ts

    flat_states = np.concatenate(collected, axis=-1)

    sep_len = int(sep) if isinstance(sep, (bool, np.bool_)) else int(sep)
    sep_tokens = 2 * np.ones((B, sep_len), dtype=np.uint8)

    return np.concatenate((initial_states, sep_tokens, flat_states), axis=-1)


def generate_all_steps(B, steps, width, rule, burnin, sep, seed):
    """Generate initial + all intermediate states (1..steps) flattened in time-major order."""
    rng = np.random.default_rng(seed)
    initial_states = rng.integers(0, 2, size=(B, width), dtype=np.uint8)
    if burnin:
        initial_states = evolve_CA(initial_states, 1000, rule)

    current = initial_states
    states = []
    for _ in range(steps):
        current = evolve_CA(current, 1, rule)
        states.append(current)
    all_states = np.stack(states, axis=1)  # (B, steps, width)
    flat_states = all_states.reshape(B, steps * width)

    sep_len = int(sep) if isinstance(sep, (bool, np.bool_)) else int(sep)
    sep_tokens = 2 * np.ones((B, sep_len), dtype=np.uint8)

    return np.concatenate((initial_states, sep_tokens, flat_states), axis=-1)
