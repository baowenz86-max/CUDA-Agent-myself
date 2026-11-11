import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pooling_kernel_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.avg_pooling = nn.AvgPool1d(pooling_kernel_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.avg_pooling(x)
        return x


batch_size = 128
num_embeddings = 100
embedding_dim = 64
height = 8192  # Increased from 512 to 8192 to increase computation
pooling_kernel_size = 3


def get_inputs():
    return [torch.randint(num_embeddings, (batch_size, height))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, pooling_kernel_size]