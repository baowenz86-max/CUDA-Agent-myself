import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.argsort(x, dim=1)
        return x


num_embeddings = 1024
embedding_dim = 64


def get_inputs():
    return [torch.randint(0, num_embeddings, (128, 2048))]


def get_init_inputs():
    return [num_embeddings, embedding_dim]