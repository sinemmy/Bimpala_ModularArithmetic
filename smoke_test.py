"""
Smoke test — verifies the full pipeline end-to-end in seconds on CPU.

Checks:
  1. Data generation and angular embedding
  2. Train/test split sizes
  3. Both models: instantiation, forward pass, backward pass
  4. A mini training loop (a few batches)

Run with:
    python smoke_test.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.data import angular_embed, make_dataset
from src.models.vanilla_lstm import VanillaLSTM
from src.models.bilinear_lstm import BilinearLSTM

P = 17
HIDDEN = 16
BATCH = 8

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}")
        return result
    except Exception as exc:
        print(f"  {FAIL}  {label}")
        raise exc


# ── 1. Angular embedding ──────────────────────────────────────────────────────
print("\n── 1. Angular embedding ─────────────────────────────────────────────")

t = torch.arange(P)
xy = check("angular_embed shape", lambda: angular_embed(t, P))
check("output is (P, 2)", lambda: None if xy.shape == (P, 2) else 1 / 0)
check(
    "values on unit circle",
    lambda: None if (xy.norm(dim=1) - 1.0).abs().max() < 1e-5 else 1 / 0,
)
check(
    "0 and P-1 are neighbors (closer than 0 and P//2)",
    lambda: None
    if (xy[0] - xy[-1]).norm() < (xy[0] - xy[P // 2]).norm()
    else 1 / 0,
)

# ── 2. Dataset ─────────────────────────────────────────────────────────────────
print("\n── 2. Dataset ───────────────────────────────────────────────────────")

train_x, train_y, test_x, test_y = check(
    "make_dataset", lambda: make_dataset(P, train_frac=0.8, seed=42)
)
n_total = P * P
n_train = int(0.8 * n_total)
n_test = n_total - n_train
check(
    f"train size = {n_train}",
    lambda: None if train_x.size(0) == n_train else 1 / 0,
)
check(
    f"test size = {n_test}",
    lambda: None if test_x.size(0) == n_test else 1 / 0,
)
check(
    "train_x shape (n, 2, 2)",
    lambda: None if train_x.shape == (n_train, 2, 2) else 1 / 0,
)
check(
    "targets in [0, P)",
    lambda: None
    if train_y.min() >= 0 and train_y.max() < P and test_y.min() >= 0 and test_y.max() < P
    else 1 / 0,
)

# ── 3. Models: forward + backward ─────────────────────────────────────────────
print("\n── 3. Models: forward + backward ────────────────────────────────────")

criterion = nn.CrossEntropyLoss()
xb = train_x[:BATCH]
yb = train_y[:BATCH]

for name, cls in [("VanillaLSTM", VanillaLSTM), ("BilinearLSTM", BilinearLSTM)]:
    model = check(f"{name} init", lambda c=cls: c(p=P, hidden_size=HIDDEN))
    logits = check(
        f"{name} forward → (batch, P)",
        lambda m=model: m(xb) if m(xb).shape == (BATCH, P) else 1 / 0,
    )
    loss = criterion(logits, yb)
    check(f"{name} backward", lambda: loss.backward())
    check(
        f"{name} grads exist",
        lambda m=model: None
        if all(p.grad is not None for p in m.parameters())
        else 1 / 0,
    )
    check(
        f"{name} grads non-zero",
        lambda m=model: None
        if all(p.grad.abs().max().item() > 1e-9 for p in m.parameters() if p.grad is not None)
        else 1 / 0,
    )

# ── 4. Mini training loop ─────────────────────────────────────────────────────
print("\n── 4. Mini training loop ────────────────────────────────────────────")

for name, cls in [("VanillaLSTM", VanillaLSTM), ("BilinearLSTM", BilinearLSTM)]:

    def _mini_train(c=cls):
        m = c(p=P, hidden_size=HIDDEN)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        m.train()
        for _ in range(5):
            logits = m(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return loss.item()

    final_loss = check(f"{name} 5 steps (loss finite)", _mini_train)
    check(
        f"{name} loss is finite: {final_loss:.4f}",
        lambda fl=final_loss: None if torch.isfinite(torch.tensor(fl)) else 1 / 0,
    )

# ── 5. BilinearLSTM b-branch init ─────────────────────────────────────────────
print("\n── 5. BilinearLSTM b-branch init ────────────────────────────────────")

from src.models.bilinear_lstm import BilinearLSTMCell

cell = BilinearLSTMCell(input_size=2, hidden_size=HIDDEN)
for branch in ("i_b", "f_b", "g_b", "o_b", "c_b"):
    layer = getattr(cell, branch)
    check(
        f"{branch}.weight == 0 (no squashing at init)",
        lambda l=layer: None if l.weight.abs().max().item() == 0.0 else 1 / 0,
    )
    check(
        f"{branch}.bias == 1 (identity modulator at init)",
        lambda l=layer: None if (l.bias - 1.0).abs().max().item() < 1e-6 else 1 / 0,
    )

print("\n── All checks passed ────────────────────────────────────────────────\n")
