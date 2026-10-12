import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_value, scale_factor, mode):
        super().__init__()
        # 保持与参考模型相同的参数结构与 state_dict 键
        self.prelu_weight = nn.Parameter(torch.full((in_channels,), float(alpha_value), dtype=torch.float32))
        self.scale_factor = float(scale_factor)
        self.mode = mode

    def forward(self, x):
        # 调用自定义 CUDA 扩展中的算子，保持计算顺序与参考一致
        x = cuda_extension.prelu_forward(x, self.prelu_weight)
        if self.mode != 'bilinear':
            raise NotImplementedError(f'Unsupported interpolation mode: {self.mode}')
        x = cuda_extension.bilinear_interpolate_forward(x, self.scale_factor)
        x = cuda_extension.argmax_dim_forward(x, 1, True)
        return x

    def debug_forward(self, x):
        # 按 checkpoints 顺序产出中间结果，调试接口不改变计算公式
        prelu_out = cuda_extension.prelu_forward(x, self.prelu_weight)
        yield 'step_001', prelu_out
        interp_out = cuda_extension.bilinear_interpolate_forward(prelu_out, self.scale_factor)
        yield 'step_002', interp_out
        argmax_out = cuda_extension.argmax_dim_forward(interp_out, 1, True)
        yield 'step_003', argmax_out
        yield 'step_004', argmax_out
