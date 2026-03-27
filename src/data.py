"""
Sparse data generation for modular arithmetic.

Reference: §3.1 of https://arxiv.org/html/2410.03569v2
"""

import torch
from einops import rearrange


def _make_sparsity_probs(N: int, sampling: str, device=None) -> torch.Tensor:
    """
    Build a normalized probability vector over z ∈ {0, …, N-1},
    where z is the number of zero entries in a generated sample.

    Args:
        N:        sequence length
        sampling: 'inv_sqrt' or 'uniform'
        device:   target torch device

    Returns:
        probs: (N,) float tensor, sums to 1
    """
    z = torch.arange(N, dtype=torch.float32, device=device)
    if sampling == "inv_sqrt":
        # f(z) ∝ 1 / sqrt(N - z + 1); larger z → more zeros → more weight
        weights = 1.0 / (N - z + 1.0).sqrt()
    elif sampling == "uniform":
        # f(z) = 1/N for all z
        weights = torch.ones(N, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}")
    return weights / weights.sum()


def generate_batch(
    N: int,
    q: int,
    batch_size: int,
    sampling: str = "inv_sqrt",
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of (a, b) pairs for modular addition: b = sum(a) mod q.

    Sampling procedure for 'inv_sqrt' and 'uniform':
      1. Draw z ~ f(z) independently per sample (number of zero entries).
      2. Set n = N - z; sample n values uniformly from {1, …, q-1}.
      3. Pad with z zeros; randomly shuffle positions.

    For 'default': draw all N entries uniformly from Z_q (baseline; no sparsity bias).

    Args:
        N:          number of elements per sample
        q:          modulus
        batch_size: number of samples
        sampling:   'inv_sqrt' | 'uniform' | 'default'
        device:     target torch device

    Returns:
        a: (batch_size, N) integer tensor with values in Z_q
        b: (batch_size,)  integer tensor, b = sum(a) mod q
    """
    if sampling == "default":
        a = torch.randint(0, q, (batch_size, N), device=device)
    else:
        probs = _make_sparsity_probs(N, sampling, device=device)

        # Sample z (number of zeros) for every element of the batch at once.
        # torch.multinomial on a 1-D weight tensor returns (batch_size,).
        z = torch.multinomial(probs, batch_size, replacement=True)  # (B,)

        # ── Build random permutations (batched randperm via argsort of noise) ──
        perms = torch.argsort(torch.rand(batch_size, N, device=device), dim=1)  # (B, N)

        # Permuted positions {0, …, z[i]-1} are designated as zeros for sample i.
        # Shapes: (1, N) vs (B, 1)  →  broadcast to (B, N)
        perm_rank = rearrange(torch.arange(N, device=device), "n -> 1 n")
        zero_in_perm = (perm_rank < rearrange(z, "b -> b 1")).long()  # (B, N)

        # Scatter zero indicators back to original (unshuffled) positions:
        # is_zero[i, perms[i, j]] = zero_in_perm[i, j]
        is_zero = torch.zeros(batch_size, N, dtype=torch.long, device=device).scatter_(
            1, perms, zero_in_perm
        )  # (B, N)

        # Nonzero entries are drawn from {1, …, q-1}; zero entries stay 0.
        vals = torch.randint(1, q, (batch_size, N), device=device)
        a = vals * (1 - is_zero)

    b = a.sum(dim=1) % q
    return a, b


def generate_dataset(
    N: int,
    q: int,
    num_samples: int,
    sampling: str = "inv_sqrt",
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-generate a full dataset as tensors.

    Args:
        N:           sequence length
        q:           modulus
        num_samples: total number of (a, b) pairs
        sampling:    'inv_sqrt' | 'uniform' | 'default'
        device:      target torch device

    Returns:
        a: (num_samples, N) integer tensor
        b: (num_samples,)  integer tensor
    """
    return generate_batch(N, q, num_samples, sampling=sampling, device=device)


def generate_test_set(
    N: int,
    q: int,
    num_samples: int = 100_000,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the held-out test set using the true distribution (uniform over Z_q^N).

    Always uses 'default' sampling and seeds with torch.manual_seed(42) for
    reproducibility, matching the evaluation protocol from the paper.

    Args:
        N:           sequence length
        q:           modulus
        num_samples: size of the test set (default 100,000)
        device:      target torch device

    Returns:
        a: (num_samples, N) integer tensor
        b: (num_samples,)  integer tensor
    """
    torch.manual_seed(42)
    return generate_batch(N, q, num_samples, sampling="default", device=device)
