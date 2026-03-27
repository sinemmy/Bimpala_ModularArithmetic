"""
Custom loss function for modular arithmetic with angular embedding.

The loss combines a standard angular MSE with a regularisation term that
prevents the model from collapsing its predictions to the origin:

    ℓ = α · (r² + 1 / (r² + ε)) + ‖(x, y) − (x′, y′)‖²

where:
    (x′, y′) = model prediction
    (x,  y)  = encode(b, q)  — ground truth on the unit circle
    r²       = x′² + y′²
    α        = 1e-4
    ε        = 1e-8  (numerical stability for the 1/r² term)

The regularisation term is minimised when r² = 1 (unit circle), so it
simultaneously discourages the origin and encourages unit-norm predictions.

Reference: §3.2 of https://arxiv.org/html/2410.03569v2
"""

import torch
from einops import reduce

from .embedding import encode

ALPHA: float = 1e-4
EPS: float = 1e-8


def modular_loss(pred_xy: torch.Tensor, b: torch.Tensor, q: int) -> torch.Tensor:
    """
    Compute the custom modular-arithmetic loss.

    Args:
        pred_xy: (batch, 2) model predictions (x′, y′)
        b:       (batch,)  ground-truth integer labels in Z_q
        q:       modulus

    Returns:
        Scalar loss tensor (mean over the batch).
    """
    target_xy = encode(b, q)  # (batch, 2)

    x_pred, y_pred = pred_xy.unbind(dim=-1)
    r2 = x_pred ** 2 + y_pred ** 2  # (batch,)

    # Per-sample angular MSE: (x − x′)² + (y − y′)²
    mse = reduce((pred_xy - target_xy) ** 2, "b two -> b", "sum")  # (batch,)

    # Origin-avoidance regularisation: α · (r² + 1/(r² + ε))
    reg = ALPHA * (r2 + 1.0 / (r2 + EPS))  # (batch,)

    return (mse + reg).mean()
