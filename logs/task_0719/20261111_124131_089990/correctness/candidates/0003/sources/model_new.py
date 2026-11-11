import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters to preserve

    def forward(self, query, key, value):
        query = cuda_extension.custom_softmax(query, dim=-1)
        output = cuda_extension.custom_scaled_dot_product_attention(query, key, value)
        return output

    def debug_forward(self, query, key, value):
        query_softmax = cuda_extension.custom_softmax(query, dim=-1)
        yield 'step_001', query_softmax
        output = cuda_extension.custom_scaled_dot_product_attention(query_softmax, key, value)
        yield 'step_002', output
        yield 'step_003', output
