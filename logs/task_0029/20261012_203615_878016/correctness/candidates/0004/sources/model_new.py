import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x_detached = cuda_extension.custom_detach(x)  # custom detach to avoid torch.detach
        unbind_tuple = cuda_extension.custom_unbind(x_detached, -1)
        max_results = []
        for xi in unbind_tuple:
            values, indices = cuda_extension.custom_max_dim(xi, 1)
            max_results.append(torch.return_types.max((values, indices)))
        return max_results

    def debug_forward(self, x):
        x_detached = cuda_extension.custom_detach(x)
        yield ('step_001', x_detached)
        unbind_tuple = cuda_extension.custom_unbind(x_detached, -1)
        yield ('step_002', unbind_tuple)
        max_results = []
        for xi in unbind_tuple:
            values, indices = cuda_extension.custom_max_dim(xi, 1)
            max_results.append(torch.return_types.max((values, indices)))
        yield ('step_003', max_results)
        yield ('step_004', max_results)
