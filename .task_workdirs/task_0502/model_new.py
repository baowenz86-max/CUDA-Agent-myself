import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = cuda_extension.matmul(x, y)
        z = cuda_extension.softmax_dim0(z)
        return z

    def debug_forward(self, x: torch.Tensor, y: torch.Tensor):
        z = cuda_extension.matmul(x, y)
        yield 'step_001', z
        z = cuda_extension.softmax_dim0(z)
        yield 'step_002', z
        yield 'step_003', z
