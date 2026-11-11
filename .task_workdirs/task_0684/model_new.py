import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        embedded = cuda_extension.custom_embedding(x, self.embedding.weight)
        split_tensors = cuda_extension.custom_split(embedded, 2, 0)
        return split_tensors

    def debug_forward(self, x):
        embedded = cuda_extension.custom_embedding(x, self.embedding.weight)
        yield "step_001", embedded
        split_tensors = cuda_extension.custom_split(embedded, 2, 0)
        yield "step_002", split_tensors
        yield "step_003", split_tensors