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
        # Custom op: cuda_extension.addmm_forward(bias, mat1, mat2)
        #   bias: [N]
        #   mat1: [M, K]
        #   mat2: [K, N]
        #   output: [M, N] = mat1 @ mat2 + bias (broadcast)
        # Here K = input_size, N = hidden_size, M = batch_size.
        # Passing self.weight directly (shape [hidden_size, input_size] = [N, K]) as mat2,
        # but we define mat2 as [K, N] -> need transpose? Actually self.weight shape is [hidden_size, input_size] = [N, K],
        # so mat2 should be [K, N] = weight.t(). To avoid transpose, we define op to accept mat2 as [N, K]
        # and compute A @ B.t()? Let's settle: reference is x @ weight.t() + bias.
        # Define addmm_forward(bias, A, B) where A is [M,K], B is [N,K], output [M,N] = A @ B.t() + bias.
        # That matches passing self.weight directly (B = weight of shape [hidden_size, input_size]).
        x = cuda_extension.addmm_forward(self.bias, x, self.weight)

        # Step 2: x = torch.unsqueeze(x, 0)
        # Custom op: cuda_extension.unsqueeze_forward(x, dim=0)
        # Returns tensor with shape [1, M, N]. Since unsqueeze is a view, but to keep interface,
        # the op returns a copy with that shape.
        x = cuda_extension.unsqueeze_forward(x, 0)

        # Step 3: x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        # Custom op: cuda_extension.interpolate_linear_forward(x, scale_factor)
        # Input x: [1, M, N] (3D, temporal). Output: [1, M, int(N*scale_factor)].
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)

        return x

    def debug_forward(self, x):
        # Same as forward but yields checkpoints in order.
        x = cuda_extension.addmm_forward(self.bias, x, self.weight)
        yield 'step_001', x
        x = cuda_extension.unsqueeze_forward(x, 0)
        yield 'step_002', x
        x = cuda_extension.interpolate_linear_forward(x, self.scale_factor)
        yield 'step_003', x
        yield 'step_004', x
