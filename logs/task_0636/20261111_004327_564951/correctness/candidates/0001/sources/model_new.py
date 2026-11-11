import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        # 保存超参数；out_channels 未参与 forward 计算，但保留签名兼容
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.maxpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        x = cuda_extension.tanh_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        step_001 = cuda_extension.maxpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        yield ('step_001', step_001)
        step_002 = cuda_extension.tanh_forward(step_001)
        yield ('step_002', step_002)
        yield ('step_003', step_002)
