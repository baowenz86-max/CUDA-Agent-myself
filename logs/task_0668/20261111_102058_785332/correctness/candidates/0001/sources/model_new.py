import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x):
        x = cuda_extension.log_softmax_dim1(x)
        x = cuda_extension.div_scalar(x, self.divisor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.log_softmax_dim1(x)
        yield "step_001", x
        x = cuda_extension.div_scalar(x, self.divisor)
        yield "step_002", x
        yield "step_003", x
