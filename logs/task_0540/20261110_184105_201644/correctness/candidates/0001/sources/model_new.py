import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = cuda_extension.batch_norm2d_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = cuda_extension.clamp_min(x, 0.1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.batch_norm2d_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.clamp_min(x, 0.1)
        yield 'step_002', x
        yield 'step_003', x
