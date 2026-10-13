import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, module_dict_items):
        super().__init__()
        # Preserve original state_dict keys: module_dict.conv1.weight, module_dict.conv1.bias, etc.
        self.module_dict = nn.ModuleDict(module_dict_items)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, key):
        # x is expected to be on CPU from task_contract, move to CUDA for custom op.
        # If already CUDA, .cuda() is a no-op if device matches.
        x = x.cuda()
        weight = self.module_dict[key].weight
        bias = self.module_dict[key].bias
        x = cuda_extension.conv_transpose1d_forward(
            x, weight, bias,
            self.stride, self.padding, self.output_padding
        )
        return x

    def debug_forward(self, *inputs):
        x, key = inputs
        x = x.cuda()
        weight = self.module_dict[key].weight
        bias = self.module_dict[key].bias
        x = cuda_extension.conv_transpose1d_forward(
            x, weight, bias,
            self.stride, self.padding, self.output_padding
        )
        yield 'step_001', x
        yield 'step_002', x
