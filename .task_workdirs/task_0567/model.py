import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.sigmoid = torch.sigmoid
        self.softplus = nn.Softplus()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.softplus(x)
        x = self.sigmoid(x)
        return x


batch_size = 128
seq_len = 64
in_features = 2048
out_features = 1024
input_data_shape = (batch_size, seq_len, in_features)


def get_inputs():
    return [torch.randn(*input_data_shape)]


def get_init_inputs():
    return [in_features, out_features]