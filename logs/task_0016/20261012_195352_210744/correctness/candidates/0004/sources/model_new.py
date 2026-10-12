import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale_factor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Step 1: x = torch.addmm(self.bias, x, self.weight.t())
        # Custom op: addmm_forward(bias, mat1, mat2) -> output
        #   bias: [hidden_size]
        #   mat1: x [batch_size, input_size]
        #   mat2: weight [hidden_size, input_size] (i.e., original weight, no transpose)
        # The kernel computes A * B directly, equivalent to A * weight.t() when B=weight
        x = cuda_extension.addmm_forward(self.bias, x, self.weight)

        # Step 2: torch.unsqueeze(x, 0)
        x = cuda_extension.unsqueeze_forward(x, 0)

        # Step 3: F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)

        return x

    def debug_forward(self, x):
        # Step 1
        x = cuda_extension.addmm_forward(self.bias, x, self.weight)
        yield 'step_001', x

        # Step 2
        x = cuda_extension.unsqueeze_forward(x, 0)
        yield 'step_002', x

        # Step 3
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)
        yield 'step_003', x

        # Final return checkpoint
        yield 'step_004', x
