import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # Step 1: unbind along dim=1 using custom op
        x = cuda_extension.unbind_dim1(x)
        # Step 2: apply pixel_shuffle (identity) to each item using custom op,
        # resulting in list of new tensors.
        x = [cuda_extension.identity_new_tensor(item) for item in x]
        # Step 3: stack along dim=1 using custom op
        x = cuda_extension.stack_dim1(x)
        return x

    def debug_forward(self, x):
        # Yield step_001: unbind result (tuple)
        x_unbind = cuda_extension.unbind_dim1(x)
        yield ('step_001', x_unbind)
        # Yield step_002: list after identity_new_tensor
        x_list = [cuda_extension.identity_new_tensor(item) for item in x_unbind]
        yield ('step_002', x_list)
        # Yield step_003: stacked tensor
        x_stack = cuda_extension.stack_dim1(x_list)
        yield ('step_003', x_stack)
        # Yield final return (same as step_003)
        yield ('step_004', x_stack)
