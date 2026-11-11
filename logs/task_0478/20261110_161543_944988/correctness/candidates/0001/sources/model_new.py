import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Step 1: softmax along dim=1. Move input to current CUDA device (execution_device=cuda)
        # The reference inputs are CPU tensors, but the model runs on CUDA.
        if not x.is_cuda:
            x = x.to('cuda')
        x = cuda_extension.custom_softmax_lastdim_v2(x)

        # Step 2: Mish activation
        x = cuda_extension.custom_mish(x)

        return x

    def debug_forward(self, x):
        # Same device handling as forward
        if not x.is_cuda:
            x = x.to('cuda')

        # checkpoint step_001: after softmax
        softmax_out = cuda_extension.custom_softmax_lastdim_v2(x)
        yield 'step_001', softmax_out

        # checkpoint step_002: after mish
        mish_out = cuda_extension.custom_mish(softmax_out)
        yield 'step_002', mish_out

        # checkpoint step_003: final return
        yield 'step_003', mish_out
