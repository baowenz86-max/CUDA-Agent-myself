import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # Keep module attributes to preserve state_dict keys and structure.
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()
        self.conv_transpose3d = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Move input to CUDA if not already, reference execution is on CUDA.
        x = x.cuda()

        # Step 1: adaptive_avg_pool2d on last two dims.
        x = cuda_extension.custom_adaptive_avg_pool2d_5d(
            x,
            output_size=1,
            output_size_y=1
        )

        # Step 2: conv3d
        x = cuda_extension.custom_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            stride=self.conv3d.stride[0],
            padding=self.conv3d.padding[0]
        )

        # Step 3: leaky_relu
        x = cuda_extension.custom_leaky_relu(
            x,
            negative_slope=self.relu.negative_slope
        )

        # Step 4: bias add
        x = cuda_extension.custom_bias_add(x, self.bias)

        # Step 5: clamp
        x = cuda_extension.custom_clamp(x, min_val=0.0, max_val=1.0)

        # Step 6: conv_transpose3d
        x = cuda_extension.custom_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            stride=self.conv_transpose3d.stride[0],
            padding=self.conv_transpose3d.padding[0],
            output_padding=self.conv_transpose3d.output_padding[0]
        )

        return x

    def debug_forward(self, *inputs):
        x = inputs[0].cuda()

        # Step 1
        x = cuda_extension.custom_adaptive_avg_pool2d_5d(
            x,
            output_size=1,
            output_size_y=1
        )
        yield 'step_001', x

        # Step 2
        x = cuda_extension.custom_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            stride=self.conv3d.stride[0],
            padding=self.conv3d.padding[0]
        )
        yield 'step_002', x

        # Step 3
        x = cuda_extension.custom_leaky_relu(
            x,
            negative_slope=self.relu.negative_slope
        )
        yield 'step_003', x

        # Step 4
        x = cuda_extension.custom_bias_add(x, self.bias)
        yield 'step_004', x

        # Step 5
        x = cuda_extension.custom_clamp(x, min_val=0.0, max_val=1.0)
        yield 'step_005', x

        # Step 6
        x = cuda_extension.custom_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            stride=self.conv_transpose3d.stride[0],
            padding=self.conv_transpose3d.padding[0],
            output_padding=self.conv_transpose3d.output_padding[0]
        )
        yield 'step_006', x

        # Step 7 (final return)
        yield 'step_007', x
