import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, eps=1e-5):
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        # No parameters/buffers are created because the reference InstanceNorm2d
        # has affine=False, track_running_stats=False, and no learnable state.
        # This keeps state_dict() empty, matching Model.

    def forward(self, x):
        x = cuda_extension.instance_norm_2d_forward(x, self.eps)
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.instance_norm_2d_forward(x, self.eps)
        yield ('step_001', x)
        x = cuda_extension.relu_forward(x)
        yield ('step_002', x)
        yield ('step_003', x)
        return x
