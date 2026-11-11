import torch
import torch.nn as nn
import cuda_extension


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

        # Use standard submodules to preserve state_dict keys
        self.linear = nn.Linear(in_features, out_features)
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        x = cuda_extension.unsqueeze_forward(x, 2)
        x = cuda_extension.interpolate_nearest_1d(x, float(self.scale_factor))
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        return x

    def debug_forward(self, x):
        """Yield (checkpoint_id, value) for each reference step."""
        # Step 1: linear
        x1 = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        yield "step_001", x1

        # Step 2: unsqueeze
        x2 = cuda_extension.unsqueeze_forward(x1, 2)
        yield "step_002", x2

        # Step 3: interpolate
        x3 = cuda_extension.interpolate_nearest_1d(x2, float(self.scale_factor))
        yield "step_003", x3

        # Step 4: conv_transpose1d
        x4 = cuda_extension.conv_transpose1d_forward(
            x3,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_004", x4

        # Step 5: return
        yield "step_005", x4
