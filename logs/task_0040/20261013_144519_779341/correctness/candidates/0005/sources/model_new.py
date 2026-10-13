import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        num_features,
        out_features,
    ):
        super().__init__()
        # 使用与参考模型相同的子模块结构，确保 state_dict 键一致
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.empty(bias_shape))
        self.linear = nn.Linear(num_features, out_features)

        # 保存构造参数以便在 forward 中使用
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.out_features = out_features

        # 初始化额外 bias 参数（参考模型中为 torch.randn）
        nn.init.normal_(self.bias)

    def forward(self, x):
        # 步骤 1：ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        # 步骤 2：添加 bias
        x = cuda_extension.add_bias_forward(x, self.bias)
        # 步骤 3：roll
        x = cuda_extension.roll_forward(x, shifts=1, dims=2)
        # 步骤 4：view
        x = cuda_extension.view_forward(x, self.out_features, self.num_features)
        # 步骤 5：线性层
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        # 按检查点顺序 yield (id, value)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x

        x = cuda_extension.add_bias_forward(x, self.bias)
        yield "step_002", x

        x = cuda_extension.roll_forward(x, shifts=1, dims=2)
        yield "step_003", x

        x = cuda_extension.view_forward(x, self.out_features, self.num_features)
        yield "step_004", x

        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        yield "step_005", x

        yield "step_006", x
