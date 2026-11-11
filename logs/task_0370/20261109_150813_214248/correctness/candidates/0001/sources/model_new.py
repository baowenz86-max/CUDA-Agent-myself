import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        # Match reference module structure to preserve state_dict keys
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x: torch.Tensor):
        # Custom GroupNorm (CUDA) replacing self.group_norm(x)
        groupnorm_x = cuda_extension.group_norm_forward(
            x, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps
        )

        # Preserve exactly the non-CUDA dummy tensor path
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        equal = torch.equal(a, b)

        return groupnorm_x, equal

    def debug_forward(self, x: torch.Tensor):
        groupnorm_x = cuda_extension.group_norm_forward(
            x, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps
        )
        yield 'step_001', groupnorm_x

        a = torch.tensor([1, 2, 3])
        yield 'step_002', a

        b = torch.tensor([1, 2, 3])
        yield 'step_003', b

        equal = torch.equal(a, b)
        yield 'step_004', equal

        yield 'step_005', (groupnorm_x, equal)
