import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, sync_bn):
        super().__init__()
        self.sync_bn = sync_bn
        self.norm = nn.SyncBatchNorm(num_channels) if sync_bn else nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = torch.view_as_real(x)
        x = self.norm(x)
        return x


batch_size = 512
num_channels = 128
seq_len = 10240  # Increased sequence length to increase computation


def get_inputs():
    real_part = torch.randn(batch_size, num_channels, seq_len)
    imag_part = torch.randn(batch_size, num_channels, seq_len)
    x = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
    return [x]


def get_init_inputs():
    num_channels = 128
    sync_bn = True
    return [num_channels, sync_bn]