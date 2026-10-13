import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.reflection_pad(x)
        return torch.sub(x, 0.5)


input_dim1 = 51200  # Increased from 512 to 51200 to increase runtime
input_dim2 = 16


def get_inputs():
    return [torch.randint(0, 5, (input_dim1, input_dim2))]


def get_init_inputs():
    return [5, 64, (3, 3)]