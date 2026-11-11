import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scalar_value):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scalar = torch.scalar_tensor(scalar_value)

    def forward(self, x):
        x = self.embedding(x)
        x = x * self.scalar
        return x


batch_size = 128
num_embeddings = 1000
embedding_dim = 256
scalar_value = 0.5
seq_length = 4096  # Increased sequence length to increase computation


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, seq_length))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, scalar_value]