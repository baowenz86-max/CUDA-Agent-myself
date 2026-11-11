import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, num_classes, loss_reduction="mean"):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.loss_reduction = loss_reduction

    def forward(self, x1, x2):
        x = cuda_extension.cat_dim1_forward(x1, x2)
        loss = cuda_extension.smooth_l1_loss_mean_forward(x)
        return loss

    def debug_forward(self, x1, x2):
        x = cuda_extension.cat_dim1_forward(x1, x2)
        yield "step_001", x
        loss = cuda_extension.smooth_l1_loss_mean_forward(x)
        yield "step_002", loss
        yield "step_003", loss


def get_init_inputs():
    return [3, 2, "mean"]


def get_inputs():
    return [torch.randn(512, 10, 128, 128), torch.randn(512, 10, 128, 128)]
