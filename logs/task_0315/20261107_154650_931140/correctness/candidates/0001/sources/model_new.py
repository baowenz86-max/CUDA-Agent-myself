import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_layer = nn.Tanh()
        self.leaky_relu_layer = nn.LeakyReLU()

    def forward(self, x):
        return cuda_extension.fused_tanh_leaky_relu(x)

    def debug_forward(self, x):
        x = cuda_extension.fused_tanh_leaky_relu(x)
        yield 'step_001', x
        x = self.leaky_relu_layer(x)
        yield 'step_002', x
        yield 'step_003', x
