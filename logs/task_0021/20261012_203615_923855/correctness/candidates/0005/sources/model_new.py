import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # Use nn.Conv1d to preserve state_dict keys 'conv1d.weight' and 'conv1d.bias'
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        # Additional bias parameter for final addition, state_dict key 'bias'
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Step 1: conv1d
        x = cuda_extension.conv1d_forward(x, self.conv1d.weight, self.conv1d.bias)
        # Step 2: batch matmul with transpose (A @ A^T)
        x = cuda_extension.batch_matmul_forward(x)
        # Step 3: softplus
        x = cuda_extension.softplus_forward(x)
        # Step 4: add bias (shape [out_channels, 1])
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        # Must yield checkpoints in same order as reference trace
        # step_001: after conv1d
        x = cuda_extension.conv1d_forward(x, self.conv1d.weight, self.conv1d.bias)
        yield 'step_001', x
        # step_002: after batch matmul
        x = cuda_extension.batch_matmul_forward(x)
        yield 'step_002', x
        # step_003: after softplus
        x = cuda_extension.softplus_forward(x)
        yield 'step_003', x
        # step_004: after bias add
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_004', x
        # step_005: final return
        yield 'step_005', x
