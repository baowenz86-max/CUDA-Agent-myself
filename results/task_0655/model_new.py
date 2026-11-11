import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, dim):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        # No parameters; state_dict remains empty like reference (unfold has no parameters)

    def forward(self, x):
        # Ensure input is CUDA float32
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        # Step 1: unfold using custom op
        kernel_h, kernel_w = self.kernel_size
        x_unfolded = cuda_extension.custom_unfold(x, kernel_h, kernel_w)
        # Step 2: sum along dim
        x_summed = cuda_extension.custom_sum_dim(x_unfolded, self.dim)
        return x_summed

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        kernel_h, kernel_w = self.kernel_size
        # checkpoint 1: after unfold
        x_unfolded = cuda_extension.custom_unfold(x, kernel_h, kernel_w)
        yield 'step_001', x_unfolded
        # checkpoint 2: after sum
        x_summed = cuda_extension.custom_sum_dim(x_unfolded, self.dim)
        yield 'step_002', x_summed
        # checkpoint 3: final return
        yield 'step_003', x_summed
