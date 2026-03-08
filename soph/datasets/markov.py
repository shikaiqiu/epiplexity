import numpy as np
from hashlib import sha3_256
import pickle
from collections import defaultdict

def split(array):
    ret = int.from_bytes(sha3_256(pickle.dumps(array.flatten().tolist())).digest())
    return ret

# def stationary_distribution(matrix):
#     original_shape = matrix.shape
#     batch_dims = original_shape[:-2]
#     n = original_shape[-1]
    
#     matrix_reshaped = matrix.reshape(-1, n, n)
#     batch_size = matrix_reshaped.shape[0]
    
#     eigenvalues = np.zeros((batch_size, n), dtype=complex)
#     eigenvectors = np.zeros((batch_size, n, n), dtype=complex)
    
#     for i in range(batch_size):
#         eigenvalues[i], eigenvectors[i] = np.linalg.eig(matrix_reshaped[i].T)
    
#     # Find eigenvalue closest to 1 for each matrix
#     idx = np.argmin(np.abs(eigenvalues - 1.0), axis=1)
#     result = np.zeros((batch_size, n))
#     for i in range(batch_size):
#         result[i] = np.real(eigenvectors[i, :, idx[i]])
#         result[i] = result[i] / np.sum(result[i])
    
#     return result.reshape(*batch_dims, n)

def stationary_distribution(matrix, num_iterations=8):
    """
    Approximates the stationary distribution of a row-stochastic matrix by iterative squaring.
    This function expects matrix to have shape (..., n, n) and returns an array of shape (..., n)
    where each row is the stationary distribution.
    
    Parameters:
      matrix: A batch of square transition matrices.
      num_iterations: Number of times to square the matrix. The effective power is 2**num_iterations.
    """
    original_shape = matrix.shape
    batch_dims = original_shape[:-2]
    n = original_shape[-1]
    matrix_reshaped = matrix.reshape(-1, n, n)
    
    # Iteratively square the matrix.
    T = matrix_reshaped.copy()
    for _ in range(num_iterations):
        T = T@T
    
    # For a stochastic matrix, T^high will be a rank-1 matrix with each row equal
    # to the stationary distribution. We take the first row of each matrix.
    result = T[:, 0, :]
    # Normalize to ensure a proper probability distribution.
    result = result / result.sum(axis=-1, keepdims=True)
    return result.reshape(*batch_dims, n)

class NGrams:
    def __init__(self, n, length=101, num_symbols=2, free_params=-1):
        """
        n: 1+the order of the markov chain, ie unigrams ->1 bigrams ->2
        length: length of contexts generated
        num_symbols: e.g. 2: [0,1,1,0,0] 3: [1,0,0,2,2,1] 
        last_token_only: all tokens except for the last will be -1
        free_params: number of transition probability vecs that vary between contexts (out of num_symbols**(n-1)) 
        """
        assert free_params==-1
        self.length = length
        self.num_symbols = num_symbols
        self.n = n - 1
        
        self.total_transitions = total_transitions = num_symbols ** (n - 1)
        self.weights = np.ones(num_symbols)
        
        if free_params == -1:
            self.free_params = total_transitions
            self.fixed_params = None
        else:
            self.free_params = min(free_params, total_transitions)
            if total_transitions > self.free_params:
                self.fixed_params = np.random.dirichlet(self.weights, size=total_transitions - self.free_params)
            else:
                self.fixed_params = None
        
        self.fixed_perm = np.random.permutation(total_transitions)
        self.powers = num_symbols ** np.arange(self.n - 1, -1, -1, dtype=np.int64)
        self.conv = np.array([num_symbols ** k for k in range(self.n)], dtype=np.int64)

    def transition_matrix_gen(self, batch_size=1):
        total = np.zeros((batch_size,self.total_transitions,self.num_symbols))
        total[:,:self.free_params,:] = np.random.dirichlet(self.weights, size=(batch_size,self.free_params))
        if self.fixed_params is not None:
            total[:,self.free_params:,:] = self.fixed_params[None]+np.zeros((batch_size,1,1))
        return total[:,self.fixed_perm,:]
    
    def compute_stationary_distribution(self, transition_matrices):
        if self.n > 1:
            batch_size = transition_matrices.shape[0]
            states = self.num_symbols ** self.n
            temp = np.zeros((batch_size, states, states))
            for i in range(states):
                for j in range(self.num_symbols):
                    converted = int((i % (self.num_symbols ** (self.n - 1))) * self.num_symbols + j)
                    temp[:, i, converted] = transition_matrices[:, i % transition_matrices.shape[1], j]
            
            return stationary_distribution(temp)
        elif self.n==1: # bigrams
            return stationary_distribution(transition_matrices)
        else: # unigrams
            return transition_matrices.squeeze(-2)

    def multi_symbol_convert(self, sequences): # (n-1 gram) -> idx
        return np.sum(sequences * self.conv[:sequences.shape[-1]], axis=-1)

    def single_symbol_convert(self, indices): # idx -> (n-1 gram)
        if self.n == 1:
            return indices[:, None]
        return (indices[:, None] // self.powers) % self.num_symbols

    def generate(self, batch_size=1, key=None, matrix=False):
        # Create a local random generator using default_rng.
        rng = np.random.default_rng(key)
        transition_matrices = self.transition_matrix_gen(batch_size)
        stationary_distributions = self.compute_stationary_distribution(transition_matrices)
        output = np.zeros((batch_size, self.length), dtype=np.int64)
        initial_states = vectorized_sample(stationary_distributions, rng)
        output[:, :self.n] = self.single_symbol_convert(initial_states)
        for ind in range(self.n, self.length):
            if self.n == 1:
                prev_states = output[:, ind - 1]
            else:
                prev_states = self.multi_symbol_convert(output[:, ind - self.n:ind]) % transition_matrices.shape[1]
            probs = transition_matrices[np.arange(batch_size), prev_states]
            next_states = vectorized_sample(probs, rng)
            output[:, ind] = next_states

        if matrix:
            return output, transition_matrices #(BS, total_transitions, num_symbols)
        return output

    def ngram_loss(self, sequence, n, matrix_rows, row_ids):
        """
        Online NLL (base e) for an n-gram (order n, using n-1 tokens as context).
        If the full context (length self.n) is in row_ids, use known probs from matrix_rows;
        otherwise, use dynamic online counts with Laplace smoothing.
        """
        import numpy as np
        assert n - 1 <= self.n, "n-1 must be <= self.n"
        seq = np.asarray(sequence)
        if seq.ndim == 1:
            seq = seq[None, :]
        batch_size, L = seq.shape

        ctx_fix, ctx_dyn = self.n, n - 1
        if L <= ctx_fix:
            raise AssertionError("Sequence length must exceed self.n")
        conv_base = self.num_symbols ** np.arange(ctx_fix, dtype=np.int64)
        conv_dyn  = self.num_symbols ** np.arange(ctx_dyn, dtype=np.int64)
        row_ids = np.asarray(row_ids)  # shape (num_known_rows,)

        total_log_sum = 0.0
        total_preds = batch_size * (L - ctx_fix)
        # Dynamic counts: one table per batch element.
        num_possible_dyn = self.num_symbols ** ctx_dyn
        free_counts = np.zeros((batch_size, num_possible_dyn, self.num_symbols), dtype=np.int64)

        for t in range(ctx_fix, L):
            # Full context for fixed lookup.
            full_ctx = seq[:, t - ctx_fix:t]  # shape (batch_size, ctx_fix)
            if ctx_fix == 1:
                full_idx = full_ctx.squeeze(axis=1)  # shape (batch_size,)
            else:
                full_idx = np.sum(full_ctx * conv_base, axis=1)  # shape (batch_size,)
            # Determine known vs. dynamic using row_ids.
            known_mask = np.isin(full_idx, row_ids)  # shape (batch_size,)
            dynamic_mask = ~known_mask
            preds = seq[:, t].astype(np.int64)  # shape (batch_size,)

            # Known branch: use matrix_rows (shape: (batch_size, len(row_ids), num_symbols)).
            if np.any(known_mask):
                batch_known = np.nonzero(known_mask)[0]
                known_idx = np.empty(len(batch_known), dtype=int)
                for j, b in enumerate(batch_known):
                    # Find index in row_ids matching full_idx[b]; assume unique.
                    known_idx[j] = int(np.where(row_ids == full_idx[b])[0][0])
                probs_known = matrix_rows[batch_known, known_idx, preds[batch_known]]
                total_log_sum += np.sum(np.log(probs_known))

            # Dynamic branch: update online counts.
            if np.any(dynamic_mask):
                dyn_batch = np.nonzero(dynamic_mask)[0]
                if ctx_dyn == 1:
                    # Squeeze to get a 1D array.
                    dyn_ctx = seq[dyn_batch, t - ctx_dyn:t].astype(np.int64).squeeze(axis=1)
                    dyn_idx = dyn_ctx
                else:
                    dyn_ctx = seq[dyn_batch, t - ctx_dyn:t]  # shape (num_dyn, ctx_dyn)
                    dyn_idx = np.sum(dyn_ctx * conv_dyn, axis=1)
                counts = free_counts[dyn_batch, dyn_idx, :]  # shape (num_dyn, num_symbols)
                totals = np.sum(counts, axis=1)              # shape (num_dyn,)
                preds_dyn = preds[dynamic_mask]
                # Laplace smoothing always on.
                probs_dyn = (counts[np.arange(len(preds_dyn)), preds_dyn] + 1) / (totals + self.num_symbols)
                total_log_sum += np.sum(np.log(probs_dyn))
                free_counts[dyn_batch, dyn_idx, preds_dyn] += 1

        return - total_log_sum / total_preds




    def avg_ngram_losses(self, row_ids):
        out = defaultdict(lambda: 0.0)
        for i in range(L:=10):
            # Generate sequences and the corresponding matrix_rows (when matrix=True, generate returns a tuple).
            seqs, matrix = self.generate(batch_size=10000//L, matrix=True)
            for n in range(1, self.n + 2):
                out[n] += (self.ngram_loss(seqs, n, matrix[:,row_ids,:], row_ids)-out[n])/(i+1)
            #out = {n: self.ngram_loss(seqs, n, matrix_rows, row_ids) for n in range(1, self.n + 2)}
        # For orders 1 to self.n+1, compute the loss using matrix_rows and the provided row_ids.
        return out


def vectorized_sample(probs, rng):
    """
    Given a probability distribution array `probs` of shape (..., K),
    returns an array of samples (with the same leading dimensions) using
    inverse transform sampling with the provided random generator `rng`.
    """
    # Compute the cumulative distribution function along the last axis.
    cdf = np.cumsum(probs, axis=-1)
    # Normalize each row by dividing by its last element.
    cdf /= cdf[..., -1:]
    # Generate uniform samples using the provided RNG.
    uniform_samples = rng.random(probs.shape[:-1])
    # The index is computed as the number of cdf entries less than the sample.
    return (cdf < uniform_samples[..., None]).sum(axis=-1)