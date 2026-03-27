"""
Strictly bilinear IMPALA-inspired modular arithmetic model (no CNN).

Identical structure to SemiBilinearModular but uses StrictlyBilinearLSTM,
which removes all sigmoid/tanh nonlinearities — including the cell-state
readout. Every gate and every activation is a bilinear product.

Architecture:
    Input  (batch, 2*N)
        ↓  rearrange → (N, batch, 2)
    BilinearGatedFC(2 → 20)         — gated per-token embedding
        ↓  (N, batch, 20)
    StrictlyBilinearLSTM(20 → 64)   — sequence over N tokens
        ↓  final hidden  (batch, 64)
    StrictlyBilinearLSTM(64 → 256)  — single step
        ↓  (batch, 256)
    Linear(256 → output_dim)
"""

import torch
import torch.nn as nn
from einops import rearrange

from .strictly_bilinear_IMPALA import BilinearGatedFC, StrictlyBilinearLSTM


class StrictlyBilinearModular(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2):
        """
        Args:
            input_dim:  2*N — the flat angular-embedded input size.
            output_dim: 2   — predicts (x', y') ∈ ℝ².
        """
        super().__init__()
        self.embed   = BilinearGatedFC(2,  20,  bias=False)
        self.lstm64  = StrictlyBilinearLSTM(input_size=20,  hidden_size=64)
        self.lstm256 = StrictlyBilinearLSTM(input_size=64,  hidden_size=256)
        self.fc_out  = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2*N) angular-embedded inputs.

        Returns:
            (batch, 2) predicted (x', y') coordinates.
        """
        # (batch, 2*N) → (N, batch, 2)
        tokens = rearrange(x, "b (n two) -> n b two", two=2)

        # Bilinear gated per-token embedding → (N, batch, 20)
        embedded = self.embed(tokens)

        # StrictlyBilinearLSTM 64 over N tokens → take final hidden (batch, 64)
        seq64, _ = self.lstm64(embedded)
        h64 = seq64[-1]

        # StrictlyBilinearLSTM 256 single step → (batch, 256)
        h256, _ = self.lstm256(h64)

        return self.fc_out(h256)  # (batch, 2)
