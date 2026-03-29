"""
Model-agnostic training script for modular arithmetic (Nanda-style).

Usage:
    python train.py --model vanilla_lstm --p 113
    python train.py --model bilinear_lstm --p 113 --hidden-size 128
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv

DEFAULT_WANDB_ENTITY  = "narmal"
DEFAULT_WANDB_PROJECT = "bilinearLSTM"


sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(Path(__file__).parent / ".env")
_wandb_key = os.environ.get("WANDB_KEY")
if _wandb_key:
    wandb.login(key=_wandb_key, relogin=False)

from src.data import make_dataset
from src.models.vanilla_lstm import VanillaLSTM
from src.models.bilinear_lstm import BilinearLSTM

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "vanilla_lstm": VanillaLSTM,
    "bilinear_lstm": BilinearLSTM,
}


def _weight_norm(model: nn.Module) -> float:
    """Total L2 norm of all parameters."""
    return math.sqrt(sum(p.data.norm().item() ** 2 for p in model.parameters()))


def train(args: argparse.Namespace):
    device = torch.device(args.device)

    # ── W&B ────────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    train_x, train_y, test_x, test_y = make_dataset(
        args.p, train_frac=args.train_frac, seed=args.seed,
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    n_train = train_x.size(0)
    print(f"Data: {n_train} train / {test_x.size(0)} test  (p={args.p})")

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cls = MODELS[args.model]
    model = model_cls(p=args.p, hidden_size=args.hidden_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  ({n_params:,} params)")
    wandb.run.summary["n_params"] = n_params

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_grad_norm = 0.0
        n_batches = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = train_x[idx], train_y[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Always compute grad norm for logging; clip only if requested
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip if args.grad_clip > 0 else float("inf"),
            ).item()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            epoch_correct += (logits.argmax(dim=1) == yb).sum().item()
            epoch_grad_norm += grad_norm
            n_batches += 1

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train
        avg_grad_norm = epoch_grad_norm / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log every-epoch metrics ────────────────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "grad_norm": avg_grad_norm,
            "lr": current_lr,
        })

        # ── Evaluate every log_every epochs ────────────────────────────────────
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x)
                test_loss = criterion(test_logits, test_y).item()
                test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

            w_norm = _weight_norm(model)

            wandb.log({
                "test/loss": test_loss,
                "test/acc": test_acc,
                "weight_norm": w_norm,
            })

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                wandb.run.summary["best_test_acc"] = best_test_acc
                wandb.run.summary["best_test_acc_epoch"] = epoch

            print(
                f"[{epoch:>6d}/{args.epochs}]  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
                f"grad_norm={avg_grad_norm:.4f}  weight_norm={w_norm:.4f}"
            )

    print(f"Done. Best test acc: {best_test_acc:.4f}")
    run.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train modular arithmetic model (Nanda-style)",
    )
    parser.add_argument("--model", type=str, default="vanilla_lstm",
                        choices=list(MODELS.keys()))
    parser.add_argument("--p", type=int, default=113,
                        help="Prime modulus")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="LSTM hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Evaluate and print metrics every N epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY,
                        help="W&B entity (username or team)")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT,
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if omitted)")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
