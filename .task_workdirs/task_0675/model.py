import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv3d = nn.Conv3d(embedding_dim, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(2)  # Add a temporal dimension
        return self.conv3d(x)


batch_size = 128
num_embeddings = 5000
embedding_dim = 128
out_channels = 128  # Increased from 64 to 128 to increase computation
kernel_size = 3
stride = 1
padding = 1
seq_len = 512  # Maintained to keep spatial dimension sufficient


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, seq_len))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, out_channels, kernel_size, stride, padding]