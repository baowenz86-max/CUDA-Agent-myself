import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x1, x2):
        mean_x1 = torch.mean(x1, dim=self.embedding_dim)
        mean_x2 = torch.mean(x2, dim=self.embedding_dim)
        return F.cosine_similarity(mean_x1, mean_x2)


batch_size = 256  # Increased from 16 to 256 to increase computation
channels = 3
height = 512
width = 256
embedding_dim = 1


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [embedding_dim]