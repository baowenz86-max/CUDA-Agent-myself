import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, kernel_size, stride):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride)

    def forward(self, x):
        x = self.embedding(x)
        x = self.avg_pool(x)
        return x


num_embeddings = 10
embedding_dim = 128  # Increased to make embedding output larger
batch_size = 16      # Increased batch size
height = 512         # Kept at increased value
width = 64           # Increased from 32 to expand input width
kernel_size = (2, 2)
stride = 2


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, height, width))
    return [x]


def get_init_inputs():
    return [num_embeddings, embedding_dim, kernel_size, stride]