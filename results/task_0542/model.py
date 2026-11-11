import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, seq_len, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, indices, batch1, batch2):
        x = self.embedding(indices)
        x = torch.baddbmm(x, batch1, batch2)
        return x


num_embeddings = 1000
embedding_dim = 128
batch_size = 512
seq_len = 512  # Increased from 32 to 512 to increase computation
input_dim = 256


def get_inputs():
    indices = torch.randint(0, num_embeddings, (batch_size, seq_len))
    batch1 = torch.randn(batch_size, seq_len, input_dim)
    batch2 = torch.randn(batch_size, input_dim, embedding_dim)
    return [indices, batch1, batch2]


def get_init_inputs():
    return [num_embeddings, embedding_dim, batch_size, seq_len, input_dim]