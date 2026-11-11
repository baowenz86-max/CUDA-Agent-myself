import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        # No parameters or buffers in reference model

    def forward(self, x):
        # Custom op performs max over dim=1 and view_as_complex
        result, _ = cuda_extension.custom_max_view_as_complex(x)
        return result

    def debug_forward(self, x):
        # Yield each checkpoint in order, matching reference_steps
        max_result, indices = cuda_extension.custom_max_view_as_complex(x)
        # step_001: tuple of max values (float32) and indices (int64), keepdim=True shape [128,1,512,512,2]
        yield "step_001", (max_result.unsqueeze(1), indices.unsqueeze(1))
        # step_002 and step_003: complex output after squeeze and view_as_complex
        complex_result = max_result.squeeze(1)
        yield "step_002", complex_result
        yield "step_003", complex_result


batch_size = 128
in_channels = 2
height, width = 512, 512


def get_inputs():
    real_part = torch.randn(batch_size, in_channels, height, width)
    imag_part = torch.randn(batch_size, in_channels, height, width)
    x = torch.stack((real_part, imag_part), dim=-1)
    return [x]


def get_init_inputs():
    return [in_channels, height, width]
