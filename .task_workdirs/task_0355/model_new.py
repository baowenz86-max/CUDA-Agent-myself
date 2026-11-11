import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.randn(num_features))
        self.bias = nn.Parameter(torch.randn(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _, _ = cuda_extension.batch_norm(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )
        return output

    def debug_forward(self, *inputs):
        x = inputs[0]
        output, running_mean, running_var = cuda_extension.batch_norm(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )
        yield 'step_001', (output, running_mean, running_var)
        yield 'step_002', output
