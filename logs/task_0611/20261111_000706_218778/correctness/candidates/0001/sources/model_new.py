import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Preserve state_dict keys: layer_norm.weight, layer_norm.bias
        self.layer_norm = nn.Module()
        self.layer_norm.weight = nn.Parameter(torch.ones(num_features))
        self.layer_norm.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Input x is assumed to be CUDA float32 tensor of shape [batch, seq, num_features]
        # Move parameters to CUDA if needed (state_dict may be on CPU)
        if self.layer_norm.weight.device != x.device:
            self.layer_norm.weight = nn.Parameter(self.layer_norm.weight.to(x.device))
            self.layer_norm.bias = nn.Parameter(self.layer_norm.bias.to(x.device))
        return cuda_extension.layer_norm_sum(x, self.layer_norm.weight, self.layer_norm.bias, self.eps)

    def debug_forward(self, x):
        result = self.forward(x)
        yield 'step_001', result
