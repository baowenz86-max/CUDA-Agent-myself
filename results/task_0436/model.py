import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, height, width, depth):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.bn = nn.BatchNorm3d(embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)  # Adjust dimensions to match BatchNorm3d input requirements
        x = self.bn(x)
        x = x.permute(0, 2, 3, 4, 1)  # Adjust dimensions back to original
        return x


num_embeddings = 1000
embedding_dim = 128
batch_size = 512  # Increased to ensure at least one dimension >=512
height = 16
width = 16
depth = 16


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, height, width, depth))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, batch_size, height, width, depth]