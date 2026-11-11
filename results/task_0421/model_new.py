import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, xor_tensor_shape):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.xor_tensor = nn.Parameter(torch.randn(xor_tensor_shape))

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.bitwise_xor_int64(x, self.xor_tensor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield 'step_001', x
        x = cuda_extension.bitwise_xor_int64(x, self.xor_tensor)
        yield 'step_002', x
        yield 'step_003', x
