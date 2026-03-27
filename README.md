# Bimpala — Modular Arithmetic

Investigating whether **bilinear-gated LSTMs** learn modular arithmetic better than ReLU baselines.

Implements the data pipeline, angular embedding, custom loss, and training loop from:
> *Making Hard Problems Easier with Custom Data Distributions and Loss Regularization: A Case Study in Modular Arithmetic* — Saxena et al., 2024 ([arXiv:2410.03569](https://arxiv.org/abs/2410.03569))

---

## Project structure

```
.
├── train.py              # Model-agnostic training script
├── smoke_test.py         # End-to-end sanity check (runs in seconds on CPU)
├── .env                  # WANDB_KEY, WANDB_ENTITY, WANDB_PROJECT (gitignored)
└── src/
    ├── data.py           # Sparse data generation (inv_sqrt / uniform / default)
    ├── embedding.py      # Angular embedding: encode / decode
    ├── loss.py           # Custom modular loss with origin-avoidance regularisation
    ├── metrics.py        # τ-accuracy and angle MSE
    └── models/
        ├── relu_modular.py             # ReLU baseline  (new, for mod arithmetic)
        ├── semi_bilinear_modular.py    # Semi-bilinear  (new, for mod arithmetic)
        ├── strictly_bilinear_modular.py# Strictly bilinear (new, for mod arithmetic)
        ├── relu_IMPALA.py              # Original IMPALA backbone (visual RL)
        ├── semi_bilinear_IMPALA.py     # Bilinear IMPALA backbone (visual RL)
        ├── strictly_bilinear_IMPALA.py # Strictly bilinear IMPALA (visual RL)
        └── obsolete/                   # Earlier CNN experiments
```

---

## Models

All three **modular arithmetic models** share the same IMPALA-inspired blue-ladder architecture (CNN dropped entirely) and differ only in their nonlinearity. Each satisfies `forward(x: Tensor[batch, 2*N]) → Tensor[batch, 2]`.

```
(batch, 2*N)  →  per-token embedding (dim 20)
                       ↓  (N, batch, 20)
              LSTM 64  — runs over all N tokens
                       ↓  final hidden (batch, 64)
              LSTM 256 — single step
                       ↓  (batch, 256)
              Linear   → (batch, 2)   # predicts (x', y') on the unit circle
```

### `ReluModular` — `src/models/relu_modular.py`
ReLU baseline. Standard sigmoid/tanh LSTM gates throughout.
- **Embedding:** `Linear(2→20) + ReLU`
- **LSTMs:** `StandardLSTM` — `nn.LSTMCell` (sigmoid/tanh)

### `SemiBilinearModular` — `src/models/semi_bilinear_modular.py`
Bilinear gating in embedding and LSTM gates; `tanh` cell-state readout retained.
- **Embedding:** `BilinearGatedFC(2→20)` — `transform(x) × gate(x)`
- **LSTMs:** `BilinearLSTM` — bilinear gates (i, f, g, o), standard `tanh(c)` readout

### `StrictlyBilinearModular` — `src/models/strictly_bilinear_modular.py`
Fully bilinear — no sigmoid or tanh anywhere. The entire network is a polynomial map.
- **Embedding:** `BilinearGatedFC(2→20)`
- **LSTMs:** `StrictlyBilinearLSTM` — bilinear gates **and** bilinear cell readout

---

## Training pipeline

### Key ideas from the paper

| Component | What we do |
|---|---|
| **Data** | Sample sparsity count *z* from `f_inv_sqrt(z) ∝ 1/√(N−z+1)`; sample *n=N−z* nonzero entries from {1,…,q−1}; pad + shuffle |
| **Embedding** | `(cos(2πt/q), sin(2πt/q))` — maps Z_q onto the unit circle |
| **Loss** | `α(r² + 1/(r²+ε)) + ‖(x,y)−(x′,y′)‖²`, α=1e-4 — MSE + origin-avoidance |
| **Metrics** | Angle MSE; τ-accuracy at τ=0.5% and τ=1% (circular distance ≤ τ·q) |

### Hyperparameters (§3.3)

| | |
|---|---|
| Optimiser | Adam, lr=3e-5 |
| LR schedule | Linear warm-up (1k steps) → cosine decay |
| Batch size | 250 |
| Distinct training samples | 10M (inv_sqrt sampling) |
| Total budget | 100M (~10 epochs) |
| Test set | 100k, uniform Z_q^N, seed=42 |

---

## Usage

### Smoke test (verify everything before cloud)
```bash
python smoke_test.py
```
Runs ~7 checks (data, embedding, loss, all three models, metrics, W&B, mini training loop) in a few seconds on CPU.

### Full training
```bash
python train.py \
  --N 16 --q 257 \
  --model src.models.relu_modular.ReluModular \
  --wandb-run-name relu-N16-q257

python train.py \
  --N 16 --q 257 \
  --model src.models.semi_bilinear_modular.SemiBilinearModular \
  --wandb-run-name semi-bilinear-N16-q257

python train.py \
  --N 16 --q 257 \
  --model src.models.strictly_bilinear_modular.StrictlyBilinearModular \
  --wandb-run-name strictly-bilinear-N16-q257
```

All flags:
```
--N                 Number of elements per sample (e.g. 16, 32, 64, 128)
--q                 Modulus (e.g. 257, 3329, 42899, 974269)
--model             Fully-qualified model class path
--device            cuda / cpu (auto-detected)
--checkpoint-dir    Where to save best checkpoint (default: checkpoints/)
--wandb-entity      W&B entity (default: narmal)
--wandb-project     W&B project (default: bilinearLSTM)
--wandb-run-name    W&B run name (auto-generated if omitted)
```

### Credentials
Add to `.env` (gitignored):
```
WANDB_KEY=your_api_key
WANDB_ENTITY=narmal        # optional override for smoke_test
WANDB_PROJECT=bilinearLSTM # optional override for smoke_test
```

---

## Vast.ai quickstart

### 1. Rent an instance
- Template: **PyTorch (Vast)** — `vastai/pytorch`
- GPU: any with ≥ 8 GB VRAM (RTX 2080 Ti / A4000 or better)
- CUDA: ≥ 12.1
- Container size: **30 GB**, no volume needed (checkpoints go to W&B)

### 2. Add your SSH key
Account → SSH Keys → paste contents of `~/.ssh/your_key.pub`

### 3. Add the instance to ~/.ssh/config (on your local machine)
```
Host vastai
    HostName <IP from Connect page>
    Port <PORT from Connect page>
    User root
    IdentityFile ~/.ssh/your_key
```
Then connect with `ssh vastai` or via VSCode Remote-SSH.

### 4. Set up the repo (on the remote, inside the auto-started tmux session)
```bash
# Vast auto-starts tmux — you're already protected from disconnects
# Check if uv is available
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

git clone https://github.com/<your-username>/Bimpala_ModularArithmetic
cd Bimpala_ModularArithmetic
uv sync
echo "WANDB_KEY=<your_key>" > .env
```

### 5. Verify and train
```bash
# Sanity check
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
uv run python smoke_test.py

# Kick off a run (stays alive if you disconnect — tmux handles it)
uv run python train.py --N 16 --q 257 --model src.models.relu_modular.ReluModular --wandb-run-name relu-N16-q257
```

### 6. Detach / reattach tmux
| Action | Keys |
|---|---|
| Detach (leave training running) | `Ctrl+B` then `D` |
| Reattach after reconnecting | `tmux attach` |

---

## Adding a new model
1. Create `src/models/my_model.py` with a class that accepts `__init__(self, input_dim, output_dim)` and implements `forward(x: Tensor[batch, input_dim]) → Tensor[batch, 2]`.
2. Pass it to the training script: `--model src.models.my_model.MyModel`.
