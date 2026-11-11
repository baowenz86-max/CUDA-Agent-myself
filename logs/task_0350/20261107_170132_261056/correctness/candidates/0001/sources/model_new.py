import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias, seq_modules):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.seq = nn.Sequential(*seq_modules)

    def _get_conv_transpose_params(self):
        return self.conv_transpose1d.weight, self.conv_transpose1d.bias, self.conv_transpose1d.stride[0], self.conv_transpose1d.padding[0], self.conv_transpose1d.output_padding[0]

    def _get_conv1d_params(self):
        conv1d = self.seq[0]
        return conv1d.weight, conv1d.bias, conv1d.stride[0], conv1d.padding[0]

    def _get_maxpool_params(self):
        maxpool = self.seq[2]
        return maxpool.kernel_size, maxpool.stride, maxpool.padding

    def forward(self, x):
        w_convtrans, b_convtrans, stride_convtrans, padding_convtrans, output_padding_convtrans = self._get_conv_transpose_params()
        x = cuda_extension.conv_transpose1d_forward(
            x, w_convtrans, b_convtrans, stride_convtrans, padding_convtrans, output_padding_convtrans
        )

        w_conv1d, b_conv1d, stride_conv1d, padding_conv1d = self._get_conv1d_params()
        x = cuda_extension.conv1d_forward(x, w_conv1d, b_conv1d, stride_conv1d, padding_conv1d)

        x = cuda_extension.relu_forward(x)

        kernel_size, stride_pool, padding_pool = self._get_maxpool_params()
        x = cuda_extension.max_pool1d_forward(x, kernel_size, stride_pool, padding_pool)

        return x

    def debug_forward(self, x):
        w_convtrans, b_convtrans, stride_convtrans, padding_convtrans, output_padding_convtrans = self._get_conv_transpose_params()
        x = cuda_extension.conv_transpose1d_forward(
            x, w_convtrans, b_convtrans, stride_convtrans, padding_convtrans, output_padding_convtrans
        )
        yield "step_001", x

        w_conv1d, b_conv1d, stride_conv1d, padding_conv1d = self._get_conv1d_params()
        x = cuda_extension.conv1d_forward(x, w_conv1d, b_conv1d, stride_conv1d, padding_conv1d)
        x = cuda_extension.relu_forward(x)

        kernel_size, stride_pool, padding_pool = self._get_maxpool_params()
        x = cuda_extension.max_pool1d_forward(x, kernel_size, stride_pool, padding_pool)
        yield "step_002", x

        yield "step_003", x
