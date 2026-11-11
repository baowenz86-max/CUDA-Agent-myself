import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return F.smooth_l1_loss(self.embedding(x), torch.randn_like(self.embedding(x)))


num_embeddings = 1000
embedding_dim = 300


def get_inputs():
    return [torch.randint(0, num_embeddings, (1024, 256))]


def get_init_inputs():
    return [num_embeddings, embedding_dim]