import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        return x


vocab_size = 1000
embedding_dim = 32
normalized_shape = (embedding_dim,)


def get_inputs():
    batch_size = 64
    sequence_len = 5120  # Increased sequence length to increase computation
    x = torch.randint(0, vocab_size, (batch_size, sequence_len))
    return [x]


def get_init_inputs():
    return [vocab_size, embedding_dim, normalized_shape]