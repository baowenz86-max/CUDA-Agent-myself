import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.log_softmax_forward(x, 1)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield 'step_001', x1
        x2 = cuda_extension.log_softmax_forward(x1, 1)
        yield 'step_002', x2
        yield 'step_003', x2
