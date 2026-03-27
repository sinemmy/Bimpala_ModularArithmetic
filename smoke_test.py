"""
Smoke test — verifies the full pipeline end-to-end in ~seconds on CPU.

Checks:
  1. All imports resolve
  2. Data generation (all three sampling modes)
  3. Angular embedding encode/decode round-trip
  4. Custom loss is a finite scalar
  5. All three modular models: instantiation, forward pass, backward pass
  6. W&B logging (init → log → finish)
  7. A miniature training loop (2 epochs × 10 batches × 8 samples)

Run with:
    python smoke_test.py
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

# ── W&B / env setup (mirrors train.py) ────────────────────────────────────────
import wandb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
_wandb_key = os.environ.get("WANDB_KEY")
if _wandb_key:
    wandb.login(key=_wandb_key, relogin=False)

# ── Project imports ────────────────────────────────────────────────────────────
from src.data import generate_batch, generate_dataset, generate_test_set
from src.embedding import decode, encode
from src.loss import modular_loss
from src.metrics import angle_mse, tau_accuracy
from src.models.relu_modular import ReluModular
from src.models.semi_bilinear_modular import SemiBilinearModular
from src.models.strictly_bilinear_modular import StrictlyBilinearModular

# ── Smoke-test hypers (tiny — runs on CPU in seconds) ─────────────────────────
N          = 4
Q          = 17
BATCH      = 8
N_BATCHES  = 10   # batches per epoch
N_EPOCHS   = 2
N_TRAIN    = N_BATCHES * BATCH
N_TEST     = 64
DEVICE     = torch.device("cpu")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}")
        return result
    except Exception as exc:
        print(f"  {FAIL}  {label}")
        raise exc


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Data generation ───────────────────────────────────────────────")

a_inv, b_inv = check("inv_sqrt sampling",
    lambda: generate_batch(N, Q, BATCH, sampling="inv_sqrt"))
check("uniform sampling",
    lambda: generate_batch(N, Q, BATCH, sampling="uniform"))
check("default sampling",
    lambda: generate_batch(N, Q, BATCH, sampling="default"))
check("b = sum(a) mod q",
    lambda: None if (b_inv == a_inv.sum(dim=1) % Q).all() else (_ for _ in ()).throw(AssertionError("b mismatch")))
check("generate_dataset",
    lambda: generate_dataset(N, Q, N_TRAIN, sampling="inv_sqrt"))
check("generate_test_set (seeded)",
    lambda: generate_test_set(N, Q, num_samples=N_TEST))

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Embedding encode / decode ─────────────────────────────────────")

t = torch.randint(0, Q, (BATCH,))
check("encode shape (batch,) → (batch, 2)",
    lambda: None if encode(t, Q).shape == (BATCH, 2) else 1/0)

a2d = torch.randint(0, Q, (BATCH, N))
check("encode shape (batch, N) → (batch, N, 2)",
    lambda: None if encode(a2d, Q).shape == (BATCH, N, 2) else 1/0)

def _round_trip():
    vals = torch.arange(Q)
    recovered = decode(encode(vals, Q), Q)
    assert (recovered == vals).all(), f"round-trip failed: {recovered}"

check("decode(encode(t)) == t for all t in Z_q", _round_trip)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. Loss ──────────────────────────────────────────────────────────")

pred_xy = torch.randn(BATCH, 2)
b_rand  = torch.randint(0, Q, (BATCH,))

def _loss_check():
    loss = modular_loss(pred_xy, b_rand, Q)
    assert loss.ndim == 0 and loss.isfinite(), f"bad loss: {loss}"
    loss.backward()   # check backward runs

check("modular_loss is finite scalar and backward runs", _loss_check)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Models — instantiation + forward + backward ───────────────────")

models = {
    "ReluModular":           ReluModular,
    "SemiBilinearModular":   SemiBilinearModular,
    "StrictlyBilinearModular": StrictlyBilinearModular,
}
x_in = torch.randn(BATCH, 2 * N)

for name, cls in models.items():
    def _model_check(cls=cls, name=name):
        m = cls(input_dim=2 * N, output_dim=2)
        out = m(x_in)
        assert out.shape == (BATCH, 2), f"{name} output shape {out.shape}"
        b_fake = torch.randint(0, Q, (BATCH,))
        loss = modular_loss(out, b_fake, Q)
        loss.backward()
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return n_p

    n_params = check(f"{name}: forward + backward ({cls.__name__})", _model_check)
    print(f"       ({n_params:,} parameters)")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. Metrics ───────────────────────────────────────────────────────")

pred_s    = torch.randint(0, Q, (N_TEST,))
target_s  = torch.randint(0, Q, (N_TEST,))
pred_xy2  = torch.randn(N_TEST, 2)
target_xy = encode(target_s, Q)

check("tau_accuracy returns float in [0,1]",
    lambda: None if 0.0 <= tau_accuracy(pred_s, target_s, Q, tau=0.01) <= 1.0 else 1/0)
check("angle_mse returns non-negative float",
    lambda: None if angle_mse(pred_xy2, target_xy) >= 0 else 1/0)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. W&B logging ───────────────────────────────────────────────────")

def _wandb_check():
    run = wandb.init(
        project="bimpala-modular-arithmetic",
        name="smoke-test",
        config={"N": N, "q": Q, "smoke": True},
        tags=["smoke-test"],
    )
    wandb.log({"batch/loss": 1.23, "batch/lr": 3e-5})
    wandb.log({"epoch/train_loss": 1.0, "epoch/mse": 0.5,
               "epoch/tau_0.5pct": 0.1, "epoch/tau_1pct": 0.2, "epoch": 1})
    wandb.run.summary["best_tau_05"] = 0.1
    run.finish()

check("wandb init → log → finish", _wandb_check)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 7. Mini training loop ────────────────────────────────────────────")

def _mini_train():
    from einops import rearrange
    import math

    model = ReluModular(input_dim=2 * N, output_dim=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-5)

    a_tr, b_tr = generate_dataset(N, Q, N_TRAIN, sampling="inv_sqrt")
    a_te, b_te = generate_test_set(N, Q, num_samples=N_TEST)

    run = wandb.init(
        project="bimpala-modular-arithmetic",
        name="smoke-mini-train",
        config={"N": N, "q": Q, "smoke": True},
        tags=["smoke-test"],
    )

    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        perm   = torch.randperm(N_TRAIN)
        a_shuf = a_tr[perm]
        b_shuf = b_tr[perm]
        for start in range(0, N_TRAIN, BATCH):
            a_b = a_shuf[start:start + BATCH].to(DEVICE)
            b_b = b_shuf[start:start + BATCH].to(DEVICE)
            x   = rearrange(encode(a_b, Q), "b n two -> b (n two)")
            out = model(x)
            loss = modular_loss(out, b_b, Q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            wandb.log({"mini/batch_loss": loss.item()})

        # eval
        model.eval()
        with torch.no_grad():
            x_te  = rearrange(encode(a_te.to(DEVICE), Q), "b n two -> b (n two)")
            p_xy  = model(x_te)
            p_s   = decode(p_xy, Q)
            t_xy  = encode(b_te.to(DEVICE), Q)
            mse   = angle_mse(p_xy, t_xy)
            acc   = tau_accuracy(p_s, b_te.to(DEVICE), Q, tau=0.01)
        wandb.log({"mini/epoch": epoch, "mini/mse": mse, "mini/tau_1pct": acc})
        model.train()

    run.finish()

check("mini training loop (2 epochs × 10 batches, ReluModular)", _mini_train)

# ══════════════════════════════════════════════════════════════════════════════
print("\n\033[92mAll checks passed — safe to move to cloud.\033[0m\n")
