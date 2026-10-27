import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self._call_linear(x)
        x = self._call_silu(x)
        x = self._call_permute(x)
        return x

    def debug_forward(self, x):
        x = self._call_linear(x)
        yield ('step_001', x)
        x = self._call_silu(x)
        yield ('step_002', x)
        x = self._call_permute(x)
        yield ('step_003', x)
        yield ('step_004', x)

    def _call_linear(self, x):
        import cuda_extension
        return cuda_extension.custom_linear_forward(x, self.linear.weight, self.linear.bias)

    def _call_silu(self, x):
        import cuda_extension
        return cuda_extension.custom_silu_forward(x)

    def _call_permute(self, x):
        import cuda_extension
        # Pass a dummy integer because the custom op hardcodes (0,2,3,1)
        return cuda_extension.custom_permute_forward(x, 0)
