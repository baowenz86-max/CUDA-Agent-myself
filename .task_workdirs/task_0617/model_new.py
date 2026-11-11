import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # No learnable parameters, matching reference state_dict (empty)
        # input_dim and hidden_dim accepted to match signature

    def forward(self, x):
        x = cuda_extension.glu_forward(x)
        return x

    def debug_forward(self, x):
        y = cuda_extension.glu_forward(x)
        yield ('step_001', y)
        yield ('step_002', y)
