import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dim):
        super().__init__()
        self.conv_transpose1d_weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.conv_transpose1d_bias = nn.Parameter(torch.empty(out_channels))
        # Store configuration as plain attributes (no torch functional modules)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dim = dim
        # Initialize like ConvTranspose1d would (kaiming_uniform for weight, uniform for bias)
        # To match state_dict keys exactly, we could instead use nn.ConvTranspose1d module but then
        # using its parameters in custom op call is fine; however, the module itself is not used for
        # computation. For simplicity and exact initialization, we use nn.ConvTranspose1d to create
        # parameters with same names and initialization.
        # We will recreate the module to get proper init and state_dict keys.
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        # Reassign parameters from conv_transpose1d to self.conv_transpose1d_weight/bias (same names)
        self.conv_transpose1d_weight.data = self.conv_transpose1d.weight.data.clone()
        self.conv_transpose1d_bias.data = self.conv_transpose1d.bias.data.clone()
        # Remove the submodule to avoid extra parameters; but we want state_dict keys conv_transpose1d.weight,
        # not conv_transpose1d_weight. We can rename parameters using nn.Module attribute naming.
        # Better: keep submodule conv_transpose1d and use its parameters directly in forward.
        # No need for separate parameters; we can just use self.conv_transpose1d.weight and .bias in forward.
        # But model_new.py must not use torch modules for computation; accessing parameters is allowed.
        # Let's simplify: keep self.conv_transpose1d as module for parameters.

    def forward(self, x):
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        x = cuda_extension.softmax(x, self.dim)
        return x

    def debug_forward(self, *inputs):
        # inputs is a tuple of tensors matching get_inputs (one tensor)
        x = inputs[0]
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        yield 'step_001', x
        x = cuda_extension.softmax(x, self.dim)
        yield 'step_002', x
        yield 'step_003', x
