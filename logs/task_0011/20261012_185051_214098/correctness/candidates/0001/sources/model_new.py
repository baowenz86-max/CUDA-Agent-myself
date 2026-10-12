import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_negative_slope, batch_norm_num_features):
        super().__init__()
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.batch_norm = nn.BatchNorm3d(num_features=batch_norm_num_features)

    def forward(self, x):
        x = cuda_extension.tanh_forward(x)
        x = cuda_extension.leaky_relu_forward(x, self.leaky_relu_negative_slope)
        x = cuda_extension.batch_norm3d_inference_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.tanh_forward(x)
        yield 'step_001', x
        x = cuda_extension.leaky_relu_forward(x, self.leaky_relu_negative_slope)
        yield 'step_002', x
        x = cuda_extension.batch_norm3d_inference_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        yield 'step_003', x
        yield 'step_004', x
