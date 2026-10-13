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
        # 初始化 ConvTranspose1d 参数（权重和偏置），保持 state_dict 键
        self.conv_transpose1d_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        self.conv_transpose1d_bias = nn.Parameter(torch.empty(out_channels))
        # 初始化额外的 bias 参数，形状为 bias_shape
        self.bias = nn.Parameter(torch.empty(bias_shape))
        # 初始化线性层参数
        self.linear_weight = nn.Parameter(torch.empty(out_features, num_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))

        # 保存构造参数以便在 forward 中使用
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.out_features = out_features

        # 参考模型中 nn.ConvTranspose1d 和 nn.Linear 的默认初始化
        # 这里手动复制相同的初始化逻辑，确保初始权重分布一致
        import math
        import torch.nn.init as init

        init.kaiming_uniform_(self.conv_transpose1d_weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv_transpose1d_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.conv_transpose1d_bias, -bound, bound)
        init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.linear_weight.size(1)) if self.linear_weight.size(1) > 0 else 0
        init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        # 步骤 1：ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d_weight,
            self.conv_transpose1d_bias,
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
        x = cuda_extension.linear_forward(x, self.linear_weight, self.linear_bias)
        return x

    def debug_forward(self, x):
        # 按检查点顺序 yield (id, value)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d_weight,
            self.conv_transpose1d_bias,
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

        x = cuda_extension.linear_forward(x, self.linear_weight, self.linear_bias)
        yield "step_005", x

        yield "step_006", x
