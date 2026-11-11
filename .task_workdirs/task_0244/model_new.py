import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Preserve original state_dict keys for batch_norm2d
        self.batch_norm2d = nn.BatchNorm2d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply tanh using custom CUDA op
        x = cuda_extension.tanh_forward(x)
        # Apply batch_norm2d using custom CUDA op
        x = cuda_extension.batch_norm2d_forward(
            x,
            self.batch_norm2d.weight,
            self.batch_norm2d.bias,
            self.batch_norm2d.running_mean,
            self.batch_norm2d.running_var,
            1e-5
        )
        return x

    def debug_forward(self, x: torch.Tensor):
        # Debug interface: yield (checkpoint_id, value) in order
        x = cuda_extension.tanh_forward(x)
        yield ('step_001', x)
        x = cuda_extension.batch_norm2d_forward(
            x,
            self.batch_norm2d.weight,
            self.batch_norm2d.bias,
            self.batch_norm2d.running_mean,
            self.batch_norm2d.running_var,
            1e-5
        )
        yield ('step_002', x)
        yield ('step_003', x)
