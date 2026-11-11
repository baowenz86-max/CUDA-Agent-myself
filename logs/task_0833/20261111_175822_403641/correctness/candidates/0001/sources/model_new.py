import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.triu = lambda x: torch.triu(x, diagonal=0)
        self.batch_norm = nn.BatchNorm2d(num_channels, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.triu_2d_forward(x, diagonal=0)
        x = cuda_extension.batch_norm2d_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.triu_2d_forward(x, diagonal=0)
        yield ("step_001", x)
        x = cuda_extension.batch_norm2d_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield ("step_002", x)
        yield ("step_003", x)
