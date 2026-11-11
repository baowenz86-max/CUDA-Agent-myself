import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Preserve exact state_dict structure of reference BatchNorm1d
        self.batch_norm = nn.Module()
        self.batch_norm.weight = nn.Parameter(torch.ones(num_features))
        self.batch_norm.bias = nn.Parameter(torch.zeros(num_features))
        self.batch_norm.register_buffer('running_mean', torch.zeros(num_features))
        self.batch_norm.register_buffer('running_var', torch.ones(num_features))
        self.batch_norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        # Move all parameters and buffers to CUDA to match execution device.
        # This ensures that the custom ops receive CUDA tensors as required by the contract.
        self.batch_norm = self.batch_norm.to('cuda')

    def forward(self, x):
        # Ensure input is on CUDA before calling custom ops.
        if not x.is_cuda:
            x = x.to('cuda')

        # Step 1: ReLU
        x = cuda_extension.relu_forward(x)
        # Step 2: BatchNorm1d in eval mode using running stats
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.eps,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.to('cuda')
        # checkpoint step_001
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        # checkpoint step_002
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.eps,
        )
        yield 'step_002', x
        # checkpoint step_003
        yield 'step_003', x
