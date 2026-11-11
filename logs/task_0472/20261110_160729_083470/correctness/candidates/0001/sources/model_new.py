import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA as expected by custom kernels
        x = x.to('cuda')
        # Reference step 1: softmax along dim=1
        x = cuda_extension.softmax_dim1_forward(x)
        # Reference step 2: relu
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        # Move input to CUDA (same as forward)
        x = x.to('cuda')
        # Step 001: softmax
        softmax_out = cuda_extension.softmax_dim1_forward(x)
        yield 'step_001', softmax_out
        # Step 002: relu applied to softmax output
        relu_out = cuda_extension.relu_forward(softmax_out)
        yield 'step_002', relu_out
        # Step 003: final return (same tensor)
        yield 'step_003', relu_out
