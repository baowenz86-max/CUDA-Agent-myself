import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, min_value):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.min_value = min_value

    def forward(self, x):
        x = self.embedding(x)
        return torch.clamp_min(x, min=self.min_value)


num_embeddings = 1000
embedding_dim = 64
min_value = 0.0


def get_inputs():
    return [torch.randint(high=num_embeddings, size=(8192, 256))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, min_value]