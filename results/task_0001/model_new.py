import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, leakyrelu_negative_slope, pow_exponent, mish_alpha,
                 relu_threshold, abs_max):
        super().__init__()
        self.leakyrelu_negative_slope = leakyrelu_negative_slope
        self.pow_exponent = pow_exponent
        self.mish_alpha = mish_alpha
        self.relu_threshold = relu_threshold
        self.abs_max = abs_max

    def forward(self, x):
        return cuda_extension.fused_activation(
            x,
            self.leakyrelu_negative_slope,
            self.pow_exponent,
            self.abs_max,
        )
