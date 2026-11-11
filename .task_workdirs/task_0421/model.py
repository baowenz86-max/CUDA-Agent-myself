import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, xor_tensor_shape):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.xor_tensor = nn.Parameter(torch.randn(xor_tensor_shape))

    def forward(self, x):
        x = self.embedding(x)
        x = torch.bitwise_xor(x.to(torch.int64), self.xor_tensor.to(torch.int64)).to(torch.float32)
        return x


batch_size = 512  # Increased from 128 to increase computation load
num_embeddings = 5000
embedding_dim = 256
sequence_length = 512
xor_tensor_shape = (sequence_length, embedding_dim)


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, sequence_length))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, xor_tensor_shape]