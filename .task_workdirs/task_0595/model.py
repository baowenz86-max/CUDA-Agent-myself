import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.equal = torch.equal
        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(out_features)

    def forward(self, x):
        # Assuming x is a 3D tensor (batch_size, in_features, seq_len)
        x = self.adaptive_avg_pool1d(x)
        return x


batch_size = 128
in_features = 128
out_features = 64
seq_len = 24576  # Increased from 512 to increase computation time


def get_inputs():
    return [torch.randn(batch_size, in_features, seq_len)]


def get_init_inputs():
    return [in_features, out_features]