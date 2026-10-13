import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, num_params):
        super().__init__()
        self.linear = nn.Linear(in_features, num_params)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_params)])

    def forward(self, input, target):
        x = cuda_extension.custom_linear(input, self.linear.weight, self.linear.bias)
        kldiv_loss = cuda_extension.custom_kldiv_loss(x, target)
        l1_loss = cuda_extension.custom_l1_loss_sum(x, self.params)
        return kldiv_loss, l1_loss

    def debug_forward(self, input, target):
        x = cuda_extension.custom_linear(input, self.linear.weight, self.linear.bias)
        yield 'step_001', x
        kldiv_loss = cuda_extension.custom_kldiv_loss(x, target)
        yield 'step_002', kldiv_loss
        l1_loss = cuda_extension.custom_l1_loss_sum(x, self.params)
        yield 'step_003', l1_loss
        result = (kldiv_loss, l1_loss)
        yield 'step_004', result
