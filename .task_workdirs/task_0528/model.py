import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.log_softmax(x, dim=1)
        return x


num_embeddings = 1000
embedding_dim = 300
padding_idx = 0


def get_inputs():
    return [torch.randint(0, num_embeddings, (16, 2048))]  # Increased sequence length to 2048


def get_init_inputs():
    return [num_embeddings, embedding_dim, padding_idx]