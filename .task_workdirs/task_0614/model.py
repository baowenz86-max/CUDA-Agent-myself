import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.gt(self.embedding(x), 0.5)


num_embeddings = 1000
embedding_dim = 300
padding_idx = 0
batch_size = 256  # Increased from 16 to 256
seq_len = 2048  # Increased from 512 to 2048


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, seq_len))
    return [x]


def get_init_inputs():
    return [num_embeddings, embedding_dim, padding_idx]