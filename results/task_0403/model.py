import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x * torch.rsqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-9)
        return x


batch_size = 128
num_embeddings = 1000
embedding_dim = 1024  # Increased from 256 to 1024 to increase computation
max_sequence_length = 512


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, max_sequence_length))]


def get_init_inputs():
    return [num_embeddings, embedding_dim]