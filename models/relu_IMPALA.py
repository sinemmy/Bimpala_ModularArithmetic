import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class StandardLSTM(nn.Module):
    """Standard LSTM layer with sigmoid/tanh gates.

    Accepts either a single frame (batch, input_size) or a sequence
    (seq_len, batch, input_size).  Hidden state tuple hx=(h, c) is optional;
    zeros are used when not supplied.
    """
    def __init__(self, input_size, hidden_size, bias=False):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size, bias=bias)

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


class reluIMPALA(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super(reluIMPALA, self).__init__()

        h, w, c = obs_space.shape
        self.num_outputs = num_outputs

        self.layer1a = ConvBlock2D(in_channels=c, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer2a = ConvBlock2D(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.layer2b = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer3a = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer4a = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flattened_dim = self._get_flattened_dim(h, w)

        self.fc1  = nn.Linear(in_features=self.flattened_dim, out_features=256, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2  = nn.Linear(in_features=256, out_features=512, bias=False)
        self.relu2 = nn.ReLU()
        self.lstm = StandardLSTM(input_size=512, hidden_size=256)

        self.fc3      = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def _get_flattened_dim(self, h, w):
        x = torch.zeros(1, 3, h, w)
        x = self.pool1(self.layer1a(x))
        x = self.pool2(self.layer2b(self.layer2a(x)))
        x = self.pool3(self.layer3a(x))
        x = self.pool4(self.layer4a(x))
        return x.numel()

    def forward(self, obs, hx=None):
        assert obs.ndim == 4
        x = obs / 255.0          # scale to 0-1
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
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x, hx = self.lstm(x, hx)   # standard LSTM: (batch, 256)

        logits = self.fc3(x)
        dist   = torch.distributions.Categorical(logits=logits)
        value  = self.value_fc(x)

        return dist, value, hx

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))

    def get_state_dict(self):
        return self.state_dict()
