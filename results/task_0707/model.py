import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = F.embedding(x, self.embedding.weight)
        x = torch.mean(x, dim=1)
        return x


batch_size = 256
seq_length = 768  # Increased to increase computation
num_embeddings = 1000
embedding_dim = 512


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, seq_length))
    return [x]


def get_init_inputs():
    return [num_embeddings, embedding_dim]