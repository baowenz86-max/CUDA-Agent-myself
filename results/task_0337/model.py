import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, height, width):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.height = height
        self.width = width

    def forward(self, indices, other_indices):
        x = self.embedding(indices)
        y = self.embedding(other_indices)
        grid_x, grid_y = torch.meshgrid(torch.arange(self.height), torch.arange(self.width))
        return x, y, grid_x, grid_y


num_embeddings = 100
embedding_dim = 128
height = 2048  # Increased to increase computational load
width = 256    # Increased to increase computational load


def get_inputs():
    indices = torch.randint(0, num_embeddings, (height, width))
    other_indices = torch.randint(0, num_embeddings, (height, width))
    return [indices, other_indices]


def get_init_inputs():
    return [num_embeddings, embedding_dim, height, width]