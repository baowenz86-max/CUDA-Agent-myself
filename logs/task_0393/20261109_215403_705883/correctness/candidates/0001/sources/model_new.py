import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

    def forward(self, log_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cuda_extension.nll_loss_forward(log_prob, target)

    def debug_forward(self, log_prob: torch.Tensor, target: torch.Tensor):
        output = cuda_extension.nll_loss_forward(log_prob, target)
        yield 'step_001', output
