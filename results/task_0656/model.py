import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, normalized_shape, eps):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = self.embedding(x)
        x = F.layer_norm(x, normalized_shape=self.layer_norm.normalized_shape, eps=self.layer_norm.eps)
        return x


num_embeddings = 1000
embedding_dim = 300
normalized_shape = (embedding_dim,)
eps = 1e-05
batch_size = 320  # Increased from 32 to 320 to increase runtime
seq_len = 512  # At least one dimension >=512


def get_inputs():
    return [torch.randint(low=0, high=num_embeddings, size=(batch_size, seq_len))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, normalized_shape, eps]