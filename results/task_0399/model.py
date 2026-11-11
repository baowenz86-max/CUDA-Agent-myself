import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, index_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.index_dim = index_dim

    def forward(self, x, indices):
        x = self.embedding(x)
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.gather(x, 1, indices)
        return x


num_embeddings = 1000
embedding_dim = 1024  # Increased from 256
batch_size = 32
sequence_length = 4096  # Increased from 512
index_dim = 4


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, sequence_length))
    indices = torch.randint(0, sequence_length, (batch_size, index_dim))
    return [x, indices]


def get_init_inputs():
    return [num_embeddings, embedding_dim, index_dim]