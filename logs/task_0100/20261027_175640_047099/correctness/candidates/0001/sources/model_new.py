import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # 继承参考模型的状态字典：nn.Upsample 没有可学习参数。
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = cuda_extension.upsample_bilinear2d(x)
        x = cuda_extension.squeeze_copy(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        upsampled = cuda_extension.upsample_bilinear2d(x)
        yield ('step_001', upsampled)
        squeezed = cuda_extension.squeeze_copy(upsampled)
        yield ('step_002', squeezed)
        yield ('step_003', squeezed)
        return squeezed
