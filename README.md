# Bilinear LSTM — Modular Arithmetic

Can a bilinear LSTM (no nonlinearities) learn modular addition?

This repo compares a **vanilla LSTM** against a **strictly bilinear LSTM** on the task of learning $(a + b) \bmod p$ for a prime $p$.

## Task

Given two integers $a, b \in \mathbb{Z}_p$, predict $c = (a + b) \bmod p$.

- **Modulus**: $p = 113$ (default)
- **Dataset**: all $p^2 = 12{,}769$ input pairs, split 80/20 into train/test
- **Output**: classification over $p$ classes via cross-entropy

## Embedding

Each integer $t \in \mathbb{Z}_p$ is mapped to a point on the unit circle:

$$t \mapsto \left(\cos\!\left(\frac{2\pi t}{p}\right),\, \sin\!\left(\frac{2\pi t}{p}\right)\right)$$

This **angular embedding** encodes the cyclic structure of $\mathbb{Z}_p$ — integers that are close mod $p$ are close on the circle (e.g. 0 and 112 are neighbors). The embedding is fixed (not learned), producing a 2D vector per token.

Each input pair $(a, b)$ becomes a 2-step sequence of shape `(2, 2)`.

## Models

Both models share the same architecture — the only difference is the LSTM cell:

```
Input (batch, 2, 2)  →  LSTM(input=2, hidden=128)  →  Linear(128, p)  →  logits
```

### Vanilla LSTM
Standard `nn.LSTM` with sigmoid gates and tanh activations.

### Bilinear LSTM
All sigmoid/tanh nonlinearities replaced with bilinear products:
- **Gates**: `[i, f, g, o] = W1([x, h]) * W2([x, h])` (element-wise product of two linear projections)
- **Cell readout**: `h = o * (W3(c) * W4(c))` instead of `h = o * tanh(c)`

No nonlinearities of any kind — the entire model is a polynomial function of its inputs.

## Training

- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Loss**: CrossEntropyLoss
- **LR schedule**: constant (no decay)
- **Batch size**: 256 (~40 batches/epoch)
- **Epochs**: 5,000 (default)

## Metrics (logged to W&B)

| Metric | Frequency |
|---|---|
| `train/loss` | every epoch |
| `train/acc` | every epoch |
| `grad_norm` | every epoch |
| `lr` | every epoch |
| `test/loss` | every 50 epochs |
| `test/acc` | every 50 epochs |
| `weight_norm` | every 50 epochs |

## Setup

```bash
pip install torch wandb python-dotenv
```

Create a `.env` file with your W&B key:
```
WANDB_KEY=your_key_here
```

## Usage

```bash
# Train vanilla LSTM
python train.py --model vanilla_lstm --p 113

# Train bilinear LSTM
python train.py --model bilinear_lstm --p 113

# All options
python train.py \
  --model vanilla_lstm \
  --p 113 \
  --hidden-size 128 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --batch-size 256 \
  --epochs 5000 \
  --train-frac 0.8 \
  --grad-clip 0.0 \
  --seed 42 \
  --log-every 50 \
  --device cuda \
  --wandb-entity your_entity \
  --wandb-project modular-arithmetic \
  --wandb-run-name my_run
```

## Cloud Quick Start (Vast.ai / Lambda / etc.)

```bash
# 1. SSH into your instance
ssh root@<your-instance-ip> -p <port>

# 2. Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# 3. Clone the repo
git clone 
git clone -b attempt2 https://github.com/sinemmy/bimpala_modulararithmetic.git
cd bimpala_modulararithmetic/Bimpala_ModularArithmetic

# 4. Create venv and install dependencies
uv sync

# 5. Set up your W&B key
echo "WANDB_KEY=your_key_here" > .env

# 6. Run smoke test
uv run python smoke_test.py

# 7. Train
uv run python train.py --model vanilla_lstm --p 113 --device cuda
uv run python train.py --model bilinear_lstm --p 113 --device cuda
```

**GPU requirements**: Any GPU with 2+ GB VRAM. The dataset is ~200KB and the model is ~67K params — this is a very lightweight workload.

## Project Structure

```
train.py                      # Training script
src/
  data.py                     # Dataset generation + angular embedding
  models/
    vanilla_lstm.py            # Standard LSTM baseline
    bilinear_lstm.py           # Strictly bilinear LSTM (no nonlinearities)
```

## References

- Nanda et al., [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (2023) — task setup, grokking phenomenon
- Bimpas et al., [Bilinear Sequence Regression](https://arxiv.org/abs/2410.03569) (2024) — bilinear LSTM architecture, angular embedding
