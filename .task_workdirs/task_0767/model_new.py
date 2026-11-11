import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.softmax_exp_forward(x, dim=1)
        return x

    def debug_forward(self, x):
        # Compute each reference step using the same underlying custom op to
        # respect the restriction against torch operators. The custom op's
        # implementation is structured to match F.softmax(x, dim=1) followed by
        # torch.exp(x); here we call separate exported helpers that compose to
        # the identical numerical result without using torch functional ops.
        softmax_out = cuda_extension.softmax_forward(x, dim=1)
        yield ('step_001', softmax_out)
        exp_out = cuda_extension.exp_forward(softmax_out)
        yield ('step_002', exp_out)
        yield ('step_003', exp_out)


batch_size = 16
height, width = 2560, 160


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []
