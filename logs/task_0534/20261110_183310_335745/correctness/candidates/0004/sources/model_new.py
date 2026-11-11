import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_values, _ = cuda_extension.max_dim_keepdim_forward(x, dim=1, keepdim=True)
        similarity = cuda_extension.cosine_similarity_dim_forward(x, max_values, dim=1)
        return similarity

    def debug_forward(self, x):
        max_values, max_indices = cuda_extension.max_dim_keepdim_forward(x, dim=1, keepdim=True)
        yield 'step_001', (max_values, max_indices)
        similarity = cuda_extension.cosine_similarity_dim_forward(x, max_values, dim=1)
        yield 'step_002', similarity
        yield 'step_003', similarity

