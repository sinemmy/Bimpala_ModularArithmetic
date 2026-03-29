"""
Data generation for modular arithmetic (Nanda-style).

Generates all p^2 pairs (a, b) with targets (a + b) mod p,
angular-embeds them, and provides a fixed train/test split.
"""

import torch


def angular_embed(t: torch.Tensor, p: int) -> torch.Tensor:
    """
    Map integers t ∈ Z_p to unit-circle coordinates.

    Args:
        t: integer tensor of arbitrary shape
        p: modulus

    Returns:
        Tensor of shape (*t.shape, 2) with [cos(2πt/p), sin(2πt/p)].
    """
    phi = 2 * torch.pi * t.float() / p
    return torch.stack([phi.cos(), phi.sin()], dim=-1)


def make_dataset(
    p: int,
    train_frac: float = 0.3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all p^2 pairs, angular-embed, and split into train/test.

    Args:
        p:          prime modulus (e.g. 113)
        train_frac: fraction of pairs used for training
        seed:       random seed for reproducible split

    Returns:
        train_x:  (n_train, 2, 2) angular-embedded input pairs
        train_y:  (n_train,)      integer targets in Z_p
        test_x:   (n_test, 2, 2)  angular-embedded input pairs
        test_y:   (n_test,)       integer targets in Z_p
    """
    a = torch.arange(p).repeat_interleave(p)  # [0,0,...,1,1,...,112,112,...]
    b = torch.arange(p).repeat(p)              # [0,1,...,112,0,1,...,112,...]
    targets = (a + b) % p

    # Angular embed both inputs → stack as 2-step sequence: (p^2, 2, 2)
    x = torch.stack([angular_embed(a, p), angular_embed(b, p)], dim=1)

    # Fixed random split
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    n_train = int(train_frac * p * p)

    train_x, train_y = x[perm[:n_train]], targets[perm[:n_train]]
    test_x, test_y = x[perm[n_train:]], targets[perm[n_train:]]

    return train_x, train_y, test_x, test_y
