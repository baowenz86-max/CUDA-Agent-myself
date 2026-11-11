import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, bias_shape):
        super().__init__()
        # weight parameter is present in reference but unused; keep for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Step 1: torch.tril(x)
        x = cuda_extension.custom_tril(x)
        # Step 2: F.elu(x)
        x = cuda_extension.custom_elu(x)
        # Step 3: x + self.bias
        x = cuda_extension.custom_add_bias(x, self.bias)
        return x

    def debug_forward(self, x):
        # Debug version yields each checkpoint value in order
        # Step 1
        x1 = cuda_extension.custom_tril(x)
        yield 'step_001', x1
        # Step 2
        x2 = cuda_extension.custom_elu(x1)
        yield 'step_002', x2
        # Step 3
        x3 = cuda_extension.custom_add_bias(x2, self.bias)
        yield 'step_003', x3
        # Step 4 (final return)
        yield 'step_004', x3
