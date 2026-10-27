import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.masked_select_forward(x, mask)
        return cuda_extension.prelu_forward(x, self.prelu.weight)

    def debug_forward(self, *inputs):
        # inputs: [x, mask]
        x, mask = inputs
        # 第一步：masked_select
        x1 = cuda_extension.masked_select_forward(x, mask)
        yield 'step_001', x1
        # 第二步：prelu（最终返回）
        x2 = cuda_extension.prelu_forward(x1, self.prelu.weight)
        yield 'step_002', x2
