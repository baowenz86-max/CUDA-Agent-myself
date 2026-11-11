import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def get_inputs():
    seq_len = 5
    batch_size = 512  # Increased to ensure at least one dimension >=512
    input_size = 10
    hidden_size = 20
    num_layers = 2
    x = torch.randn(seq_len, batch_size, input_size)
    return [x]


def get_init_inputs():
    seq_len = 5
    batch_size = 16  # Unused in Model init, kept as original
    input_size = 10
    hidden_size = 200  # Increased to increase computation
    num_layers = 2
    out_size = 1
    return [input_size, hidden_size, out_size, num_layers]