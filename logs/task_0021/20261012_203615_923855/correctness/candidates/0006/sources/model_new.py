import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv1d_forward(x, self.conv1d.weight, self.conv1d.bias)
        x = cuda_extension.batch_matmul_forward(x, x.transpose(1, 2))
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv1d_forward(x, self.conv1d.weight, self.conv1d.bias)
        yield "step_001", x
        x = cuda_extension.batch_matmul_forward(x, x.transpose(1, 2))
        yield "step_002", x
        x = cuda_extension.softplus_forward(x)
        yield "step_003", x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield "step_004", x
        yield "step_005", x
