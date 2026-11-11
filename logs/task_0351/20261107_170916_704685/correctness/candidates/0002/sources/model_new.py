import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scalar_value):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scalar = torch.scalar_tensor(scalar_value)

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.mul_scalar(x, self.scalar)
        return x

    def debug_forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield ('step_001', x)
        x = cuda_extension.mul_scalar(x, self.scalar)
        yield ('step_002', x)
        yield ('step_003', x)
        return
