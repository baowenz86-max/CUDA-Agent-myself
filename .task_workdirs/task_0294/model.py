import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        return F.embedding(x, self.embedding.weight, self.embedding.padding_idx)


num_embeddings = 1000
embedding_dim = 2048
padding_idx = 0
seq_length = 1024
batch_size = 64  # Added batch size to increase input volume


def get_inputs():
    # Generate input tensor with (batch_size, seq_length) shape
    return [torch.randint(0, num_embeddings, (batch_size, seq_length), dtype=torch.long)]


def get_init_inputs():
    return [num_embeddings, embedding_dim, padding_idx]