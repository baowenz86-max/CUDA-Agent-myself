import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, apply_sigmoid):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        x = cuda_extension.sum_dim1_keepdim(x)
        if self.apply_sigmoid:
            x = cuda_extension.sigmoid_elementwise(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.sum_dim1_keepdim(x)
        if self.apply_sigmoid:
            x = cuda_extension.sigmoid_elementwise(x)
        return x
