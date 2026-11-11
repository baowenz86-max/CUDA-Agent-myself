import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        x = cuda_extension.custom_sigmoid(x)
        x = cuda_extension.custom_instance_norm2d(x, self.eps)
        return x

    def debug_forward(self, x):
        sigmoid_out = cuda_extension.custom_sigmoid(x)
        yield 'step_001', sigmoid_out
        norm_out = cuda_extension.custom_instance_norm2d(sigmoid_out, self.eps)
        yield 'step_002', norm_out
        yield 'step_003', norm_out
