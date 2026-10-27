import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.masked_select(x, mask)
        return self.prelu(x)

    def debug_forward(self, *inputs):
        # inputs: [x, mask]
        x, mask = inputs
        # 第一步：masked_select
        x1 = torch.masked_select(x, mask)
        yield 'step_001', x1
        # 第二步：prelu（最终返回）
        x2 = self.prelu(x1)
        yield 'step_002', x2
