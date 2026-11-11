import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, hard_shrink_lambd):
        super().__init__()
        self.hard_shrink_lambd = hard_shrink_lambd
        # 未使用 out_channels，但保留参数以保持签名一致
        # 不创建 nn.MaxPool2d 模块，因为池化将通过自定义扩展实现，且 state_dict 需要为空
        # 存储 kernel_size、stride、padding 供 forward 使用
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = cuda_extension.hardshrink_forward(x, self.hard_shrink_lambd)
        x = cuda_extension.max_pool2d_forward(
            x, self.kernel_size, self.stride, self.padding
        )
        return x

    def debug_forward(self, x):
        x_shrink = cuda_extension.hardshrink_forward(x, self.hard_shrink_lambd)
        yield "step_001", x_shrink
        x_pool = cuda_extension.max_pool2d_forward(
            x_shrink, self.kernel_size, self.stride, self.padding
        )
        yield "step_002", x_pool
        yield "step_003", x_pool
