import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, log_dim):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.log_dim = log_dim

    def forward(self, x):
        x = torch.ops.cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        x = torch.ops.cuda_extension.logsumexp_dim_forward(x, self.log_dim, True)
        return x

    def debug_forward(self, x):
        x = torch.ops.cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        yield ('step_001', x)
        x = torch.ops.cuda_extension.logsumexp_dim_forward(x, self.log_dim, True)
        yield ('step_002', x)
        yield ('step_003', x)
