import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, index_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.index_dim = index_dim

    def forward(self, x, indices):
        x = cuda_extension.custom_embedding(x, self.embedding.weight)
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = cuda_extension.custom_gather(x, indices, 1)
        return x

    def debug_forward(self, *inputs):
        x, indices = inputs
        x = cuda_extension.custom_embedding(x, self.embedding.weight)
        yield 'step_001', x
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        yield 'step_002', indices
        x = cuda_extension.custom_gather(x, indices, 1)
        yield 'step_003', x
        yield 'step_004', x
        return
