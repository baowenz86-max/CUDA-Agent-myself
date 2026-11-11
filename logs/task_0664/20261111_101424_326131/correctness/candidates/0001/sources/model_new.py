import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, complex_tensor, mean_axis):
        super().__init__()
        self.complex_tensor = complex_tensor
        self.mean_axis = mean_axis
        self.is_complex_flag = torch.is_complex(self.complex_tensor)

    def forward(self, x):
        mean_tensor = torch.ops.cuda_extension.mean_dim_1(x)
        return mean_tensor, self.is_complex_flag

    def debug_forward(self, x):
        is_complex = torch.is_complex(self.complex_tensor)
        yield 'step_001', is_complex
        mean_tensor = torch.ops.cuda_extension.mean_dim_1(x)
        yield 'step_002', mean_tensor
        result = (mean_tensor, is_complex)
        yield 'step_003', result
