import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, num_params):
        super().__init__()
        self.linear = nn.Linear(in_features, num_params)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_params)])

    def forward(self, input, target):
        x = cuda_extension.linear_forward(input, self.linear.weight, self.linear.bias)
        kldiv_loss = cuda_extension.kl_div_batchmean_forward(
            cuda_extension.log_softmax_forward(x, 1),
            cuda_extension.softmax_forward(target, 1)
        )
        l1_loss = cuda_extension.l1_loss_sum_forward(x, self.params)
        return kldiv_loss, l1_loss

    def debug_forward(self, input, target):
        x = cuda_extension.linear_forward(input, self.linear.weight, self.linear.bias)
        yield 'step_001', x
        log_softmax = cuda_extension.log_softmax_forward(x, 1)
        softmax = cuda_extension.softmax_forward(target, 1)
        kldiv_loss = cuda_extension.kl_div_batchmean_forward(log_softmax, softmax)
        yield 'step_002', kldiv_loss
        l1_loss = cuda_extension.l1_loss_sum_forward(x, self.params)
        yield 'step_003', l1_loss
        result = (kldiv_loss, l1_loss)
        yield 'step_004', result
