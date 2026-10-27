import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, tensor, mean):
        super().__init__()
        # Preserve reference attribute semantics; do not register as buffers/parameters
        self.tensor = torch.tensor(tensor)
        self.mean = mean

    def forward(self, x):
        tril_out = cuda_extension.tril_forward(x, int(self.tensor.item()))
        mean_out = cuda_extension.mean_forward(x, dims=[2, 3], keepdim=True)
        return tril_out + self.mean * mean_out

    def debug_forward(self, *inputs):
        x = inputs[0]
        tril_out = cuda_extension.tril_forward(x, int(self.tensor.item()))
        mean_out = cuda_extension.mean_forward(x, dims=[2, 3], keepdim=True)
        result = tril_out + self.mean * mean_out
        yield 'step_001', result
