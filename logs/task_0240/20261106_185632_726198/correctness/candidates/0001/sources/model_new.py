import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return cuda_extension.bce_with_logits_forward(y_pred, y_true)

    def debug_forward(self, *inputs):
        y_pred, y_true = inputs
        out = cuda_extension.bce_with_logits_forward(y_pred, y_true)
        yield "step_001", out
