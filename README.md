# Bimpala_ModularArithmetic
Modular Arithmetic for BIMPALA

## Models

All models share the same convolutional backbone (bilinear gated or ReLU conv blocks with AvgPool2d) and the same pipeline: `conv layers → fc layers → LSTM → policy/value heads`. They differ only in their nonlinearity.

### `semi_bilinearCNN` — `models/semi_bilinear_IMPALA.py`
Bilinear gating throughout the conv and FC blocks, with a partially bilinear LSTM. The `tanh` on the cell-state readout is retained for stability.

- **Conv blocks:** `BilinearGatedActivation2D` — two parallel conv branches multiplied element-wise
- **FC blocks:** `BilinearGatedFC` — two parallel linear branches multiplied element-wise
- **LSTM:** `BilinearLSTM` — bilinear gates (i, f, g, o), standard `tanh(c)` readoutx
- **Pipeline:** `flatten → gatedfc1(→256) → gatedfc2(→512) → lstm(→256) → fc3 / value_fc`

### `strictlyBilinearCNN` — `models/strictly_bilinear_IMPALA.py`
Identical to `bilinearCNN` but with the `tanh` cell-state readout also replaced by a bilinear projection. The full network is a polynomial map with no classical nonlinearities, making it globally amenable to tensor/spectral decomposition.

- **Conv blocks:** `BilinearGatedActivation2D`
- **FC blocks:** `BilinearGatedFC`
- **LSTM:** `StrictlyBilinearLSTM` — bilinear gates **and** bilinear cell readout (`cell_transform(c') * cell_gate(c')`)
- **Pipeline:** `flatten → gatedfc1(→256) → gatedfc2(→512) → lstm(→256) → fc3 / value_fc`

### `reluIMPALA` — `models/relu_IMPALA.py`
Baseline counterpart to the bilinear models. All bilinear gating is replaced with ReLU activations and the LSTM uses standard sigmoid/tanh gates. Uses the same AvgPool2d as the bilinear models (not MaxPool).

- **Conv blocks:** `ConvBlock2D` — single conv + ReLU
- **FC blocks:** `nn.Linear` + ReLU
- **LSTM:** `StandardLSTM` — wraps `nn.LSTMCell` (sigmoid/tanh gates)
- **Pipeline:** `flatten → fc1+relu(→256) → fc2+relu(→512) → lstm(→256) → fc3 / value_fc`

---

All models return `(dist, value, hx)` where `hx = (h, c)` is the LSTM hidden state to be carried across rollout steps and reset at episode boundaries.
