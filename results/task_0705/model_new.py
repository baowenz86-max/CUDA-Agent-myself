import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = cuda_extension.argmax_dim(x, self.dim)
        max_result = torch.return_types.max((values, indices))
        indices_float = cuda_extension.float_cast(max_result.indices)
        x = cuda_extension.batch_norm_eval(
            indices_float,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        return x

    def debug_forward(self, x: torch.Tensor):
        values, indices = cuda_extension.argmax_dim(x, self.dim)
        max_result = torch.return_types.max((values, indices))
        yield ('step_001', max_result)
        indices_float = cuda_extension.float_cast(max_result.indices)
        out = cuda_extension.batch_norm_eval(
            indices_float,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        yield ('step_002', out)
        yield ('step_003', out)
