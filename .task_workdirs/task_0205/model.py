import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size, mask_value):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)
        self.mask_value = mask_value

    def forward(self, x):
        x = self.avg_pool(x)
        mask = torch.tensor(self.mask_value, dtype=torch.bool, device=x.device)
        return torch.masked_select(x, mask)


batch_size = 64
input_size = 512
seq_length = 10240  # Increased from 1024 to increase computation
output_size = 50
mask_value = 0.5


def get_inputs():
    x = torch.randn(batch_size, input_size, seq_length)
    return [x]


def get_init_inputs():
    return [input_size, output_size, mask_value]