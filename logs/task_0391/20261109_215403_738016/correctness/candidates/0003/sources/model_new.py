import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, mask_shape, scatter_dim):
        super().__init__()
        self.mask_shape = mask_shape
        self.scatter_dim = scatter_dim

    def forward(self, mask, input_tensor, scatter_values):
        mask = cuda_extension.expand_mask(mask)
        return cuda_extension.masked_scatter(input_tensor, mask, scatter_values)

    def debug_forward(self, mask, input_tensor, scatter_values):
        mask = cuda_extension.expand_mask(mask)
        yield ("step_001", mask)
        out = cuda_extension.masked_scatter(input_tensor, mask, scatter_values)
        yield ("step_002", out)
