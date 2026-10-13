import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, num_params):
        super().__init__()
        self.linear = nn.Linear(in_features, num_params)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_params)])

    def forward(self, input, target):
        x = cuda_extension.linear_forward(input, self.linear.weight, self.linear.bias)
        log_probs = cuda_extension.log_softmax_forward(x, 1)
        probs = cuda_extension.softmax_forward(target, 1)
        kldiv_loss = cuda_extension.kl_div_loss_forward(log_probs, probs)
        l1_loss = sum(cuda_extension.l1_loss_forward(x[:, i], self.params[i].expand_as(x[:, i])) for i in range(len(self.params)))
        return kldiv_loss, l1_loss

    def debug_forward(self, input, target):
        x = cuda_extension.linear_forward(input, self.linear.weight, self.linear.bias)
        yield 'step_001', x
        log_probs = cuda_extension.log_softmax_forward(x, 1)
        probs = cuda_extension.softmax_forward(target, 1)
        kldiv_loss = cuda_extension.kl_div_loss_forward(log_probs, probs)
        yield 'step_002', kldiv_loss
        l1_loss = sum(cuda_extension.l1_loss_forward(x[:, i], self.params[i].expand_as(x[:, i])) for i in range(len(self.params)))
        yield 'step_003', l1_loss
        result = (kldiv_loss, l1_loss)
        yield 'step_004', result
