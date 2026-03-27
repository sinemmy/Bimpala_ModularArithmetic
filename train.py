"""
Model-agnostic training script for modular arithmetic.

Reference: §3.3 of https://arxiv.org/html/2410.03569v2

Usage:
    python train.py --N 16 --q 257 --model src.models.my_model.MyModel

The model class must:
  - Accept keyword arguments  input_dim=2*N  and  output_dim=2  in its constructor.
  - Implement  forward(x: Tensor[batch, 2*N]) -> Tensor[batch, 2].
"""

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm

# Ensure the project root is on sys.path so `src.*` imports resolve correctly
# regardless of the working directory the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

# Load WANDB_KEY (and any other secrets) from .env before anything else.
load_dotenv(Path(__file__).parent / ".env")
_wandb_key = os.environ.get("WANDB_KEY")
if _wandb_key:
    wandb.login(key=_wandb_key, relogin=False)

DEFAULT_WANDB_ENTITY  = "narmal"
DEFAULT_WANDB_PROJECT = "bilinearLSTM"

from src.data import generate_dataset, generate_test_set
from src.embedding import decode, encode
from src.loss import modular_loss
from src.metrics import angle_mse, tau_accuracy

# ── Hyperparameters from §3.3 ──────────────────────────────────────────────────
BATCH_SIZE       = 250
DISTINCT_SAMPLES = 10_000_000   # 10 M unique training pairs
TOTAL_BUDGET     = 100_000_000  # 100 M total sample budget  →  10 epochs
LR               = 3e-5
WARMUP_STEPS     = 1_000
TEST_SIZE        = 100_000


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_model_cls(dotted_path: str) -> type:
    """
    Import a model class given a fully-qualified dotted path.

    Example:
        'src.models.my_model.MyModel'
        → importlib.import_module('src.models.my_model').MyModel
    """
    module_path, cls_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _embed(a: torch.Tensor, q: int) -> torch.Tensor:
    """
    Angular-embed integer input tensor.

    Args:
        a: (batch, N) integers in Z_q
        q: modulus

    Returns:
        (batch, 2*N) float tensor — the flat angular encoding of each token.
    """
    return rearrange(encode(a, q), "b n two -> b (n two)")


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    a_test: torch.Tensor,
    b_test: torch.Tensor,
    q: int,
    device: torch.device,
) -> dict[str, float]:
    """Run the full test set through the model and return all metrics."""
    model.eval()
    a_dev = a_test.to(device)
    b_dev = b_test.to(device)

    pred_xy   = model(_embed(a_dev, q))     # (test_size, 2)
    pred_s    = decode(pred_xy, q)           # (test_size,)
    target_xy = encode(b_dev, q)             # (test_size, 2)

    metrics = {
        "mse":     angle_mse(pred_xy, target_xy),
        "tau_05":  tau_accuracy(pred_s, b_dev, q, tau=0.005),
        "tau_1":   tau_accuracy(pred_s, b_dev, q, tau=0.01),
    }
    model.train()
    return metrics


# ── Main training routine ──────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> nn.Module:
    device = torch.device(args.device)

    # ── W&B run ────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity=args.wandb_entity,     # None → your default wandb account
        project=args.wandb_project,
        name=args.wandb_run_name,     # None → wandb auto-generates a name
        config={
            "N":               args.N,
            "q":               args.q,
            "model":           args.model,
            "batch_size":      BATCH_SIZE,
            "distinct_samples": DISTINCT_SAMPLES,
            "total_budget":    TOTAL_BUDGET,
            "lr":              LR,
            "warmup_steps":    WARMUP_STEPS,
            "sampling":        "inv_sqrt",
            "device":          args.device,
        },
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cls = _load_model_cls(args.model)
    model = model_cls(input_dim=2 * args.N, output_dim=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  ({n_params:,} parameters)")
    wandb.run.summary["n_params"] = n_params

    # ── Data ───────────────────────────────────────────────────────────────────
    print(
        f"Generating {DISTINCT_SAMPLES:,} training samples "
        f"(inv_sqrt, N={args.N}, q={args.q})…"
    )
    a_train, b_train = generate_dataset(
        args.N, args.q, DISTINCT_SAMPLES, sampling="inv_sqrt"
    )

    print(f"Generating {TEST_SIZE:,} test samples (default sampling, seed=42)…")
    a_test, b_test = generate_test_set(args.N, args.q, num_samples=TEST_SIZE)

    # ── Optimiser & LR schedule ────────────────────────────────────────────────
    epochs          = TOTAL_BUDGET // DISTINCT_SAMPLES           # 10
    steps_per_epoch = math.ceil(DISTINCT_SAMPLES / BATCH_SIZE)   # 40,000
    total_steps     = epochs * steps_per_epoch                   # 400,000

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Linear warm-up: lr scales from LR * 1e-3 → LR over WARMUP_STEPS steps.
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )
    # Cosine decay: LR → 0 over the remaining steps.
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - WARMUP_STEPS,
        eta_min=0.0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_STEPS],
    )

    # ── Checkpoint dir ─────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_tau05 = -1.0

    # ── Loop ───────────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, epochs + 1):
        # Re-shuffle the pre-generated dataset at the start of each epoch.
        perm   = torch.randperm(DISTINCT_SAMPLES)
        a_shuf = a_train[perm]
        b_shuf = b_train[perm]

        running_loss = 0.0
        n_batches    = 0

        pbar = tqdm(
            range(0, DISTINCT_SAMPLES, BATCH_SIZE),
            desc=f"Epoch {epoch:02d}/{epochs}",
            unit="batch",
            leave=False,
        )
        for start in pbar:
            a_b = a_shuf[start : start + BATCH_SIZE].to(device)
            b_b = b_shuf[start : start + BATCH_SIZE].to(device)

            x       = _embed(a_b, args.q)       # (batch, 2*N)
            pred_xy = model(x)                   # (batch, 2)
            loss    = modular_loss(pred_xy, b_b, args.q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches    += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )
            wandb.log({"batch/loss": loss.item(), "batch/lr": current_lr})

        avg_loss = running_loss / n_batches

        # ── Epoch-level evaluation ──────────────────────────────────────────────
        metrics = _evaluate(model, a_test, b_test, args.q, device)
        print(
            f"[Epoch {epoch:02d}/{epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"MSE={metrics['mse']:.4f}  "
            f"τ=0.5%={metrics['tau_05']:.2%}  "
            f"τ=1%={metrics['tau_1']:.2%}"
        )
        wandb.log({
            "epoch":            epoch,
            "epoch/train_loss": avg_loss,
            "epoch/mse":        metrics["mse"],
            "epoch/tau_0.5pct": metrics["tau_05"],
            "epoch/tau_1pct":   metrics["tau_1"],
        })

        # ── Save best checkpoint ────────────────────────────────────────────────
        if metrics["tau_05"] > best_tau05:
            best_tau05 = metrics["tau_05"]
            ckpt_path  = ckpt_dir / f"best_N{args.N}_q{args.q}.pt"
            torch.save(
                {
                    "epoch":       epoch,
                    "model_cls":   args.model,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "metrics":     metrics,
                    "args":        vars(args),
                },
                ckpt_path,
            )
            print(
                f"  → Checkpoint saved: {ckpt_path}  "
                f"(τ=0.5%={best_tau05:.2%})"
            )
            wandb.run.summary["best_tau_05"]    = best_tau05
            wandb.run.summary["best_epoch"]     = epoch
            wandb.run.summary["best_mse"]       = metrics["mse"]
            wandb.run.summary["best_tau_1pct"]  = metrics["tau_1"]
            wandb.save(str(ckpt_path))

    print(f"\nTraining complete.  Best τ=0.5% accuracy: {best_tau05:.2%}")
    run.finish()
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a modular arithmetic model (arxiv 2410.03569)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--N", type=int, required=True,
        help="Number of elements per sample (e.g. 16, 32, 64, 128).",
    )
    parser.add_argument(
        "--q", type=int, required=True,
        help="Modulus (e.g. 257, 3329, 42899, 974269).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=(
            "Fully-qualified import path to the model class. "
            "The class must accept keyword args (input_dim, output_dim) and "
            "implement forward(x: Tensor[batch, 2*N]) -> Tensor[batch, 2]. "
            "Example: 'src.models.my_model.MyModel'"
        ),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory in which to save the best checkpoint.",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY,
        help="W&B entity (username or team name). Defaults to your wandb account.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT,
        help="W&B project name to log runs under.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (auto-generated by wandb if omitted).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
