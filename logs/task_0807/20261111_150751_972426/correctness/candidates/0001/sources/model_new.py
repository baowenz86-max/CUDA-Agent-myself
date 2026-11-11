import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return cuda_extension.fused_relu_cos_scale(x, float(self.out_features))

    def debug_forward(self, x):
        # step_001: ReLU (inplace=False in our implementation to preserve input)
        x1 = cuda_extension.fused_relu_cos_scale(x, 0.0)  # temporary: we need only relu
        yield 'step_001', x1
        # step_002: cos + 0.1*out_features
        x2 = cuda_extension.fused_relu_cos_scale(x1, float(self.out_features))
        yield 'step_002', x2
        # step_003: return
        yield 'step_003', x2
