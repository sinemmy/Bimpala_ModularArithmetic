"""
Angular (unit-circle) embedding for modular arithmetic.

Each integer t ∈ Z_q is mapped to a point on the unit circle:
    (cos(2πt/q), sin(2πt/q))

Reference: §3.2 of https://arxiv.org/html/2410.03569v2
"""

import torch
from einops import rearrange


def encode(t: torch.Tensor, q: int) -> torch.Tensor:
    """
    Map integer tensor t ∈ Z_q to unit-circle coordinates.

    Args:
        t: integer tensor of arbitrary shape (...).
           e.g. (batch,) for labels  →  output (batch, 2)
               (batch, N) for inputs →  output (batch, N, 2)
        q: modulus

    Returns:
        Tensor of shape (*t.shape, 2) with last dim = [cos(2πt/q), sin(2πt/q)].
    """
    phi = 2 * torch.pi * t.float() / q
    # Stack cos and sin along a leading axis, then move it to the last position
    # so the output shape is (*t.shape, 2).
    return rearrange(torch.stack([phi.cos(), phi.sin()]), "two ... -> ... two")


def decode(xy: torch.Tensor, q: int) -> torch.Tensor:
    """
    Decode model output (x', y') to the nearest integer in Z_q.

    Computes φ' = atan2(y', x'), then:
        s' = round(φ' · q / (2π)) mod q

    Args:
        xy: (..., 2) tensor of (x', y') predictions
        q:  modulus

    Returns:
        Integer tensor of shape (...) with predicted values in Z_q.
    """
    x, y = xy.unbind(dim=-1)
    phi = torch.atan2(y, x)  # range (-π, π]
    s = (phi * q / (2 * torch.pi)).round().long() % q
    return s
