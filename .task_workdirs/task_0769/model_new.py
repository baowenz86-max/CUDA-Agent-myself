import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)  # Retain module for state_dict compatibility? Reference has no state_dict, but keeping attribute may be harmless; however parameterless module affects state_dict keys? It has no parameters, so state_dict remains empty. To exactly match state_dict keys, we can skip adding module entirely.

    def forward(self, x):
        # x is expected to be a list of tensors as per reference
        x = cuda_extension.concat_forward(x, dim=1)
        x = cuda_extension.softmax_forward(x, dim=1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.concat_forward(x, dim=1)
        yield 'step_001', x
        x = cuda_extension.softmax_forward(x, dim=1)
        yield 'step_002', x
        yield 'step_003', x
