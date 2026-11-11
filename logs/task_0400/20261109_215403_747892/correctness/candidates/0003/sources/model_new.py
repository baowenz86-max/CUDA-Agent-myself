import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Ensure input is on CUDA as required by execution device
        if not x.is_cuda:
            x = x.cuda()
        # Apply max reduction over dimension 1 (channel), keepdim=True
        max_values, max_indices = cuda_extension.max_reduce(x, dim=1, keepdim=True)
        # Apply ReLU6 to the max values
        relu6_out = cuda_extension.relu6(max_values)
        return relu6_out

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        # Step 001: torch.max returns (values, indices)
        max_values, max_indices = cuda_extension.max_reduce(x, dim=1, keepdim=True)
        step_001 = torch.return_types.max((max_values, max_indices))
        yield 'step_001', step_001
        # Step 002: ReLU6
        relu6_out = cuda_extension.relu6(max_values)
        yield 'step_002', relu6_out
        # Step 003: final return
        yield 'step_003', relu6_out
