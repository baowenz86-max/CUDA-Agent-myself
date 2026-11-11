import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight, self.padding_idx)
        x = cuda_extension.log_softmax_forward(x, 1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = cuda_extension.embedding_forward(x, self.embedding.weight, self.padding_idx)
        yield 'step_001', y
        z = cuda_extension.log_softmax_forward(y, 1)
        yield 'step_002', z
        yield 'step_003', z
