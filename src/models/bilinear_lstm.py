"""
Strictly bilinear LSTM for modular arithmetic.

Same architecture as VanillaLSTM but all sigmoid/tanh nonlinearities
are replaced with bilinear products: f(x) = W1(x) * W2(x).

Input:  (batch, 2, 2)  — 2-step sequence of angular-embedded tokens
Output: (batch, p)     — logits over Z_p
"""

import torch
import torch.nn as nn


class BilinearLSTMCell(nn.Module):
    """LSTM cell with all gates as bilinear products (no sigmoid/tanh).

    Standard LSTM:
        i = sigmoid(W_i[x,h])
        f = sigmoid(W_f[x,h])
        g = tanh(W_g[x,h])
        o = sigmoid(W_o[x,h])
        c = f*c + i*g
        h = o * tanh(c)

    Bilinear LSTM:
        i = W_ia([x,h]) * W_ib([x,h])
        f = W_fa([x,h]) * W_fb([x,h])
        g = W_ga([x,h]) * W_gb([x,h])
        o = W_oa([x,h]) * W_ob([x,h])
        c = f*c + i*g
        h = o * (W_ca(c) * W_cb(c))
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        # Each gate has its own bilinear pair
        self.i_a = nn.Linear(combined, hidden_size, bias=False)
        self.i_b = nn.Linear(combined, hidden_size, bias=False)
        self.f_a = nn.Linear(combined, hidden_size, bias=False)
        self.f_b = nn.Linear(combined, hidden_size, bias=False)
        self.g_a = nn.Linear(combined, hidden_size, bias=False)
        self.g_b = nn.Linear(combined, hidden_size, bias=False)
        self.o_a = nn.Linear(combined, hidden_size, bias=False)
        self.o_b = nn.Linear(combined, hidden_size, bias=False)
        # Bilinear readout of cell state (replaces tanh(c))
        self.c_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_b = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = hc
        combined = torch.cat([x, h], dim=1)
        i = self.i_a(combined) * self.i_b(combined)
        f = self.f_a(combined) * self.f_b(combined)
        g = self.g_a(combined) * self.g_b(combined)
        o = self.o_a(combined) * self.o_b(combined)
        c_new = f * c + i * g
        h_new = o * (self.c_a(c_new) * self.c_b(c_new))
        return h_new, c_new


class BilinearLSTM(nn.Module):
    def __init__(self, p: int, hidden_size: int = 128):
        """
        Args:
            p:           modulus (number of output classes)
            hidden_size: LSTM hidden dimension
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = BilinearLSTMCell(input_size=2, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 2) angular-embedded input sequence

        Returns:
            (batch, p) logits
        """
        batch = x.size(0)
        device = x.device
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)

        # Process the 2-token sequence
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], (h, c))

        return self.fc(h)
