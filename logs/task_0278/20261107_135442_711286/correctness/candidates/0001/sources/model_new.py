import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, pool_size):
        super().__init__()
        # 保持构造函数签名与原模型一致，但实际不创建任何参数
        self.in_features = in_features
        self.pool_size = pool_size
        # 记录池化参数供 forward 使用（kernel_size=pool_size，其余默认）
        self.pool_kernel = pool_size
        self.pool_stride = pool_size
        self.pool_padding = (0, 0)
        self.pool_dilation = (1, 1)
        self.pool_ceil_mode = False

    def forward(self, x):
        # 将输入移动到 CUDA（参考执行设备为 cuda）
        x = x.to('cuda')
        # 自定义 ELU 算子
        x = cuda_extension.custom_elu_forward(x)
        # 自定义 MaxPool2d 算子
        x = cuda_extension.custom_max_pool2d_forward(x)
        return x

    def debug_forward(self, *inputs):
        # 输入与 forward 相同，但按 checkpoints 逐步返回
        x = inputs[0]
        x = x.to('cuda')
        x = cuda_extension.custom_elu_forward(x)
        yield 'step_001', x
        x = cuda_extension.custom_max_pool2d_forward(x)
        yield 'step_002', x
        yield 'step_003', x
