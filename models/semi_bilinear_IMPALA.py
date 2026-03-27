import torch
import torch.nn as nn

class BilinearGatedFC(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(BilinearGatedFC, self).__init__()
        # Transform linear operation
        self.transform = nn.Linear(in_features, out_features, bias=bias)
        # Gating linear operation
        self.gate = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        transformed = self.transform(x)
        gating_factor = self.gate(x)
        return transformed * gating_factor

class BilinearLSTMCell(nn.Module):
    """Single-step LSTM cell where each gate uses bilinear gating
    (element-wise product of two linear projections) instead of sigmoid/tanh."""
    def __init__(self, input_size, hidden_size, bias=False):
        super(BilinearLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # All four gates (i, f, g, o) packed into one projection pair for efficiency
        self.transform = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.gate     = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, x, hx):
        h, c = hx
        combined  = torch.cat([x, h], dim=1)          # (batch, input+hidden)
        bilinear  = self.transform(combined) * self.gate(combined)
        i, f, g, o = bilinear.chunk(4, dim=1)         # each: (batch, hidden)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class BilinearLSTM(nn.Module):
    """Bilinear LSTM layer wrapping BilinearLSTMCell.

    Accepts either a single frame (batch, input_size) or a sequence
    (seq_len, batch, input_size).  Hidden state tuple hx=(h, c) is optional;
    zeros are used when not supplied.
    """
    def __init__(self, input_size, hidden_size, bias=False):
        super(BilinearLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = BilinearLSTMCell(input_size, hidden_size, bias=bias)

    def _init_hidden(self, batch, device):
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        return h, c

    def forward(self, x, hx=None):
        if x.dim() == 2:  # single step: (batch, input_size)
            if hx is None:
                hx = self._init_hidden(x.size(0), x.device)
            h, c = self.cell(x, hx)
            return h, (h, c)
        else:             # sequence: (seq_len, batch, input_size)
            seq_len, batch = x.size(0), x.size(1)
            if hx is None:
                hx = self._init_hidden(batch, x.device)
            h, c = hx
            outputs = []
            for t in range(seq_len):
                h, c = self.cell(x[t], (h, c))
                outputs.append(h)
            return torch.stack(outputs), (h, c)


class BilinearGatedActivation2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,padding, bias = False):
        super(BilinearGatedActivation2D, self).__init__()
        # Convolution to transform the input
        self.transform = nn.Conv2d(in_channels, out_channels, kernel_size =kernel_size, padding=padding, bias=bias) 
        # Convolution to create a gating signal
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size,padding = padding, bias=bias)

    def forward(self, x):

        transformed = self.transform(x)
        gating_factor = self.gate(x)
        return transformed * gating_factor  # Element-wise multiplication

class bilinearCNN(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super(bilinearCNN, self).__init__()

        h, w, c = obs_space.shape
        self.num_outputs = num_outputs

        self.layer1a = BilinearGatedActivation2D(in_channels=c, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer2a = BilinearGatedActivation2D(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.layer2b = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer3a = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer4a = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Compute the flattened dimension after convolutions and pooling
        self.flattened_dim = self._get_flattened_dim(h, w)

        self.gatedfc1 = BilinearGatedFC(in_features=self.flattened_dim, out_features=256)
        self.gatedfc2 = BilinearGatedFC(in_features=256, out_features=512)
        self.lstm     = BilinearLSTM(input_size=512, hidden_size=256)
        
        self.fc3 = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        

    def _get_flattened_dim(self, h, w):
        x = torch.zeros(1, 3, h, w)  # Dummy input to compute the shape
        x = self.pool1(self.layer1a(x))
        x = self.pool2(self.layer2b(self.layer2a(x)))
        x = self.pool3(self.layer3a(x))
        x = self.pool4(self.layer4a(x))
        return x.numel()

    def forward(self, obs, hx=None):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        x = self.layer1a(x)
        x = self.pool1(x)

        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.pool2(x)

        x = self.layer3a(x)
        x = self.pool3(x)

        x = self.layer4a(x)
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=1)
        x = self.gatedfc1(x)
        x = self.gatedfc2(x)
        x, hx = self.lstm(x, hx)   # bilinear LSTM: (batch, 256)
        
        logits = self.fc3(x)
        dist = torch.distributions.Categorical(logits=logits)   
        value = self.value_fc(x)
        
        return dist, value, hx

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))

    def get_state_dict(self):
        return self.state_dict()
