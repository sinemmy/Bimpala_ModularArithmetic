"""
ReLU IMPALA-inspired modular arithmetic model (no CNN).

Architecture (blue-ladder branch only):
    Input  (batch, 2*N)         — flat angular-embedded tokens
        ↓  rearrange → (N, batch, 2)
    Linear(2 → 20) + ReLU       — per-token embedding
        ↓  (N, batch, 20)
    StandardLSTM(20 → 64)       — sequence over N tokens
        ↓  final hidden  (batch, 64)
    StandardLSTM(64 → 256)      — single step
        ↓  (batch, 256)
    Linear(256 → output_dim)    — predict (x', y') on the unit circle
"""

import torch
import torch.nn as nn
from einops import rearrange

from .relu_IMPALA import StandardLSTM


class ReluModular(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2):
        """
        Args:
            input_dim:  2*N — the flat angular-embedded input size.
            output_dim: 2   — predicts (x', y') ∈ ℝ².
        """
        super().__init__()
        self.embed   = nn.Linear(2, 20, bias=False)
        self.relu    = nn.ReLU()
        self.lstm64  = StandardLSTM(input_size=20,  hidden_size=64)
        self.lstm256 = StandardLSTM(input_size=64,  hidden_size=256)
        self.fc_out  = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2*N) angular-embedded inputs.

        Returns:
            (batch, 2) predicted (x', y') coordinates.
        """
        # (batch, 2*N) → (N, batch, 2): treat each token as a sequence step
        tokens = rearrange(x, "b (n two) -> n b two", two=2)

        # Per-token linear embedding + ReLU → (N, batch, 20)
        embedded = self.relu(self.embed(tokens))

        # LSTM 64 over the full N-token sequence → (N, batch, 64)
        seq64, _ = self.lstm64(embedded)
        h64 = seq64[-1]          # final hidden state: (batch, 64)

        # LSTM 256 single step on the aggregated representation → (batch, 256)
        h256, _ = self.lstm256(h64)

        return self.fc_out(h256)  # (batch, 2)
