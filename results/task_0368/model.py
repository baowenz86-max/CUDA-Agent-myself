import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = F.log_softmax(x, dim=1)
        return x


num_embeddings = 1000
embedding_dim = 512
seq_length = 1024
batch_size = 32


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, seq_length), dtype=torch.long)]


def get_init_inputs():
    return [num_embeddings, embedding_dim]