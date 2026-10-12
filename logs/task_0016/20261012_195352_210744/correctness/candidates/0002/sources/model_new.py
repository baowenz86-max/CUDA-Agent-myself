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
        # Custom op: addmm_forward(input, mat1, mat2) -> output
        #   input: bias [hidden_size]
        #   mat1: x [batch_size, input_size]
        #   mat2: weight.t() [input_size, hidden_size]
        # 这里约定 addmm_forward 接收三个张量，返回 [batch_size, hidden_size]
        x = cuda_extension.addmm_forward(self.bias, x, self.weight.t())

        # Step 2: x = torch.unsqueeze(x, 0)
        # Custom op: unsqueeze_forward(input, dim) -> output
        x = cuda_extension.unsqueeze_forward(x, 0)

        # Step 3: x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        # Custom op: interpolate_linear_forward(input, scale_factor) -> output
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)

        return x

    def debug_forward(self, x):
        # Step 1
        x = cuda_extension.addmm_forward(self.bias, x, self.weight.t())
        yield 'step_001', x

        # Step 2
        x = cuda_extension.unsqueeze_forward(x, 0)
        yield 'step_002', x

        # Step 3
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)
        yield 'step_003', x

        # Final return checkpoint
        yield 'step_004', x
