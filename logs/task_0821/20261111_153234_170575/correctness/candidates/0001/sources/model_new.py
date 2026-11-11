import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, pad_size, pad_mode):
        super().__init__()
        # 参考模型中没有可学习参数，state_dict 应为空。
        # 此处不创建任何参数，保持与参考一致。
        self.channels = channels
        self.pad_size = pad_size
        self.pad_mode = pad_mode

    def forward(self, x):
        # 使用自定义扩展算子完成 reflection padding 和 argsort。
        # 禁止使用 torch 算子进行这些计算。
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.pad_size[0], self.pad_size[1],
                                                          self.pad_size[2], self.pad_size[3])
        x_sorted = cuda_extension.argsort_forward(x_padded, 1)
        return x_sorted

    def debug_forward(self, x):
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.pad_size[0], self.pad_size[1],
                                                          self.pad_size[2], self.pad_size[3])
        yield 'step_001', x_padded
        x_sorted = cuda_extension.argsort_forward(x_padded, 1)
        yield 'step_002', x_sorted
        yield 'step_003', x_sorted
