import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, threshold_value, out_features):
        super().__init__()
        self.threshold_value = threshold_value
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.threshold_forward(x, self.threshold_value, 0.0)

    def debug_forward(self, *inputs):
        x = inputs[0]
        output = cuda_extension.threshold_forward(x, self.threshold_value, 0.0)
        yield 'step_001', output
