import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # 参考模型没有参数，无需保存 input_channels

    def forward(self, x):
        # 输入 x 来自 get_inputs，位于 CPU，需要转移到 CUDA
        # 参考模型在 CUDA 上执行，因此这里显式转移
        x = x.to('cuda')
        x = cuda_extension.custom_elu(x, 1.0)
        x = cuda_extension.custom_relu(x)
        return x

    def debug_forward(self, x):
        # 与 forward 相同，输出各检查点
        x = x.to('cuda')
        x = cuda_extension.custom_elu(x, 1.0)
        yield 'step_001', x
        x = cuda_extension.custom_relu(x)
        yield 'step_002', x
        yield 'step_003', x
