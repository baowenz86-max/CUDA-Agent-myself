import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # 参考模型的 conv_transpose 模块虽然定义了 bias，但 forward 使用额外的 self.bias，
        # 因此这里仅初始化 conv_transpose.weight，并保留 self.bias 以匹配 state_dict。
        self.conv_transpose_weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        # 与参考模型一致的额外 bias
        self.bias = nn.Parameter(torch.empty(bias_shape))
        # 存储标量参数供 forward 使用
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # 模拟参考模型中的 nn.ConvTranspose1d，其自身也有一个未使用的 bias 参数，
        # 为保证 state_dict 键完全一致，需要添加 conv_transpose.bias。
        # 参考模型 state_dict 包含 conv_transpose.bias，因此这里保留一个 parameter。
        self.conv_transpose_bias = nn.Parameter(torch.empty(out_channels))

        # 初始化参数（与参考模型相同分布，但实际训练/测试会 load state_dict 覆盖）
        nn.init.kaiming_uniform_(self.conv_transpose_weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_transpose_bias)

    def forward(self, x):
        # 第一步：转置卷积，使用自定义算子，传入权重、额外 bias 和标量参数
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        # 第二步：torch.min(x, 1.0)
        x = cuda_extension.min_scalar_forward(x, 1.0)
        # 第三步：torch.erf(x)
        x = cuda_extension.erf_forward(x)
        return x

    def debug_forward(self, x):
        # 与 forward 完全相同的计算，每步 yield (id, value)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        yield "step_001", x
        x = cuda_extension.min_scalar_forward(x, 1.0)
        yield "step_002", x
        x = cuda_extension.erf_forward(x)
        yield "step_003", x
        yield "step_004", x
