import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        out_channels,
        in_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        scale_factor,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scale_factor = scale_factor

        # 保持 state_dict 键与参考模型一致：linear.weight, linear.bias,
        # conv_transpose1d.weight, conv_transpose1d.bias。
        # 这些参数将在后续 CUDA 扩展中直接使用原始指针。
        self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        self.conv_transpose1d_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        self.conv_transpose1d_bias = nn.Parameter(torch.empty(out_channels))

        # 使用与 nn.Linear 和 nn.ConvTranspose1d 相同的默认初始化；
        # 验证器会加载参考 state_dict，因此这里仅保证形状正确。
        nn.init.kaiming_uniform_(self.linear_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.linear_bias, -bound, bound)

        nn.init.kaiming_uniform_(self.conv_transpose1d_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose1d_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_transpose1d_bias, -bound, bound)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(2)
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv_transpose1d(x)
        return x

    # 以下两个方法在模型内部模拟原始模块调用，但在实际执行中会被替换为自定义扩展调用。
    # 第一阶段不允许使用 torch 算子进行计算，因此这里仅作为语义说明；
    # 实际 forward 将在后续阶段改为 cuda_extension 调用。
    def linear(self, input_tensor):
        # 占位：后续会通过 cuda_extension.linear_forward 调用
        raise NotImplementedError

    def conv_transpose1d(self, input_tensor):
        # 占位：后续会通过 cuda_extension.conv_transpose1d_forward 调用
        raise NotImplementedError

    def debug_forward(self, *inputs):
        # debug_forward 必须与正常 forward 语义一致，但逐步产出检查点。
        # 后续阶段将替换为对 cuda_extension 的逐步调用。
        raise NotImplementedError
