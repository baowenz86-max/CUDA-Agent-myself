import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        return self.batch_norm(x)


num_features = 64
batch_size = 128
seq_length = 16384  # Increased from 1024 to increase computation


def get_inputs():
    return [torch.randn(batch_size, num_features, seq_length)]


def get_init_inputs():
    return [num_features]