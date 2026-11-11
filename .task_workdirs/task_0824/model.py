import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, kernel_size, stride, padding, dilation):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.max_pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous()  # Change shape from [B, C, H, W] to [B, H, C, W]
        x = x.view(x.size(0), x.size(1), -1)  # Change shape to [B, H, C*W]
        x = self.max_pool(x)
        x = x.view(x.size(0), x.size(1), -1, 1)  # Change shape back to [B, H, C, 1]
        x = x.permute(0, 2, 1, 3).contiguous()  # Change shape back to [B, C, H, 1]
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
num_features = 3
height = 128  # Increased from 32 to increase computation
width = 128   # Increased from 32 to increase computation
kernel_size = 3
stride = 2
padding = 1
dilation = 1


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features, kernel_size, stride, padding, dilation]