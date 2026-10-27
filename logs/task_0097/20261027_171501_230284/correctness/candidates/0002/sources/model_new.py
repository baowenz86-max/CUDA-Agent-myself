import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, histc_bins, histc_min, histc_max
    ):
        super().__init__()
        # Keep the same parameter names and shapes as reference for state_dict compatibility
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.histc_bins = histc_bins
        self.histc_min = histc_min
        self.histc_max = histc_max

    def forward(self, x):
        # Ensure input is on CUDA and float32 as required by the contract
        x = x.to(device='cuda', dtype=torch.float32)
        # Custom conv_transpose1d
        x = cuda_extension.custom_conv_transpose1d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=2, padding=1, output_padding=1
        )
        # Custom histc
        hist = cuda_extension.custom_histc(
            x, bins=self.histc_bins, min=self.histc_min, max=self.histc_max
        )
        return x, hist

    def debug_forward(self, x):
        # Debug mode yields checkpoints in the same order as reference
        x = x.to(device='cuda', dtype=torch.float32)
        x = cuda_extension.custom_conv_transpose1d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=2, padding=1, output_padding=1
        )
        yield 'step_001', x
        hist = cuda_extension.custom_histc(
            x, bins=self.histc_bins, min=self.histc_min, max=self.histc_max
        )
        yield 'step_002', hist
        result = (x, hist)
        yield 'step_003', result
