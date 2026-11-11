import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        kd, kh, kw = self.kernel_size
        # AvgPool3d 的 stride 默认等于 kernel_size，padding 默认为 0
        return cuda_extension.avg_pool3d_forward(
            x, kd, kh, kw, kd, kh, kw, 0, 0, 0
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        kd, kh, kw = self.kernel_size
        result = cuda_extension.avg_pool3d_forward(
            x, kd, kh, kw, kd, kh, kw, 0, 0, 0
        )
        yield "step_001", result
        yield "step_002", result
