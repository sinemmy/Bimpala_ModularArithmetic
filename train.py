"""
Model-agnostic training script for modular arithmetic (Nanda-style).

Usage:
    python train.py --model vanilla_lstm --p 113
    python train.py --model bilinear_lstm --p 113 --hidden-size 128
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.data import make_dataset
from src.models.vanilla_lstm import VanillaLSTM

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "vanilla_lstm": VanillaLSTM,
    # "bilinear_lstm": BilinearLSTM,  # add later
}


def train(args: argparse.Namespace):
    device = torch.device(args.device)

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

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_correct = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = train_x[idx], train_y[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        # ── Evaluate ───────────────────────────────────────────────────────────
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x)
                test_loss = criterion(test_logits, test_y).item()
                test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

            print(
                f"[{epoch:>6d}/{args.epochs}]  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
            )

    print("Done.")


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
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print metrics every N epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
