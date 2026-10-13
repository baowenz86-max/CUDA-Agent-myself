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
        # 保持与参考模型完全相同的参数结构，以保证 state_dict 键一致
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.linear = nn.Linear(num_features, out_features)

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x = cuda_extension.bias_add_forward(x, self.bias)
        x = cuda_extension.roll_forward(x, 1, 2)
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield "step_001", x
        x = cuda_extension.bias_add_forward(x, self.bias)
        yield "step_002", x
        x = cuda_extension.roll_forward(x, 1, 2)
        yield "step_003", x
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        yield "step_004", x.view(x.size(0), -1)
        yield "step_005", x
        yield "step_006", x
        return x
