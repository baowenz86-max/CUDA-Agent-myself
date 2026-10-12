import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_embeddings, embedding_dim
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x, indices):
        x = self.conv_transpose1d(x)
        x = torch.sign(x)
        embedded = self.embedding(indices)
        return x, embedded


batch_size = 2048
in_channels = 3
out_channels = 16
length = 256
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
num_embeddings = 100
embedding_dim = 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    indices = torch.randint(0, num_embeddings, (batch_size, length))
    return [x, indices]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, num_embeddings, embedding_dim]