import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.adaptive_avg_pool(x)
        return x


num_embeddings = 1000
embedding_dim = 256  # Increased from 64 to increase computation
output_size = (1, 1)
batch_size = 64  # Increased from 32
seq_length = 512
feature_dim = 32  # Added new dimension to make input 3D for 4D embedding output


def get_inputs():
    return [torch.randint(num_embeddings, (batch_size, seq_length, feature_dim))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, output_size]