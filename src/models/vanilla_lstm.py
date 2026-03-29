"""
Minimal vanilla LSTM for modular arithmetic.

Input:  (batch, 2, 2)  — 2-step sequence of angular-embedded tokens
Output: (batch, p)     — logits over Z_p
"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    def __init__(self, p: int, hidden_size: int = 128):
        """
        Args:
            p:           modulus (number of output classes)
            hidden_size: LSTM hidden dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 2) angular-embedded input sequence

        Returns:
            (batch, p) logits
        """
        # LSTM over the 2-token sequence → take final hidden state
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden_size)
        return self.fc(h_n.squeeze(0))
