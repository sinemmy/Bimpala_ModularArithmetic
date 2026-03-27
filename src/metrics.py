"""
Evaluation metrics for modular arithmetic models.

Two metrics from §3.3 of https://arxiv.org/html/2410.03569v2:

    τ-accuracy — fraction of predictions within τ·q of the target
                 (measured on the circular / modular number line).

    angle_mse  — mean squared error between predicted and ground-truth
                 unit-circle coordinates.
"""

import torch
from einops import reduce


def tau_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    q: int,
    tau: float,
) -> float:
    """
    Fraction of samples where the circular distance to the target is ≤ τ·q.

    Circular distance:  min(|s′ − s|, q − |s′ − s|)

    Args:
        pred:   (batch,) integer predictions in Z_q
        target: (batch,) integer ground truth in Z_q
        q:      modulus
        tau:    tolerance expressed as a fraction of q (e.g. 0.005 for 0.5 %)

    Returns:
        Accuracy as a Python float in [0, 1].
    """
    diff = (pred - target).abs()
    circ_diff = torch.minimum(diff, q - diff)
    return (circ_diff <= tau * q).float().mean().item()


def angle_mse(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    """
    Mean squared error between predicted and ground-truth angular coordinates.

        MSE = mean over batch of  (x − x′)² + (y − y′)²

    Args:
        pred_xy:   (batch, 2) predicted (x′, y′) values
        target_xy: (batch, 2) ground-truth (x, y) = encode(b, q) values

    Returns:
        MSE as a Python float.
    """
    # Sum the squared error over the 2 coordinates per sample, then average.
    per_sample = reduce((pred_xy - target_xy) ** 2, "b two -> b", "sum")
    return per_sample.mean().item()
