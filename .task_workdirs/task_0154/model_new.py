import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        n_fft,
        win_length,
        hop_length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        index_dim,
        indices,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.index_dim = index_dim
        self.indices = nn.Parameter(torch.tensor(indices), requires_grad=False)

        # 保存卷积权重和偏置，但 forward 中不直接使用 torch 模块
        self.conv2d_weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.conv2d_bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # 初始化参数，与 nn.Conv2d 默认初始化一致（参考模型未指定初始化方式，使用默认）
        # 此处使用与 nn.Conv2d 相同的初始化方法，确保 state_dict 兼容
        nn.init.kaiming_uniform_(self.conv2d_weight, a=math.sqrt(5))
        if self.conv2d_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv2d_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv2d_bias, -bound, bound)

    def forward(self, x):
        # 步骤1: torch.stft
        x_stft = cuda_extension.stft_forward(
            x,
            self.n_fft,
            self.win_length,
            self.hop_length,
        )
        # 步骤2: view_as_real + permute
        x_permuted = cuda_extension.view_as_real_permute_forward(x_stft)
        # 步骤3: conv2d
        x_conv = cuda_extension.conv2d_forward(
            x_permuted,
            self.conv2d_weight,
            self.conv2d_bias,
            self.stride,
            self.padding,
            self.dilation,
        )
        # 步骤4: index_select
        x_out = cuda_extension.index_select_forward(
            x_conv,
            self.index_dim,
            self.indices,
        )
        return x_out

    def debug_forward(self, x):
        # 按检查点顺序生成中间结果和最终返回
        x_stft = cuda_extension.stft_forward(
            x,
            self.n_fft,
            self.win_length,
            self.hop_length,
        )
        yield "step_001", x_stft

        x_permuted = cuda_extension.view_as_real_permute_forward(x_stft)
        yield "step_002", x_permuted

        x_conv = cuda_extension.conv2d_forward(
            x_permuted,
            self.conv2d_weight,
            self.conv2d_bias,
            self.stride,
            self.padding,
            self.dilation,
        )
        yield "step_003", x_conv

        x_out = cuda_extension.index_select_forward(
            x_conv,
            self.index_dim,
            self.indices,
        )
        yield "step_004", x_out

        yield "step_005", x_out
