import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings_1, num_embeddings_2, embedding_dim):
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings_1, embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings_2, embedding_dim)

    def forward(self, x1, x2):
        return self.embedding1(x1) + self.embedding2(x2)


num_embeddings_1 = 1024
num_embeddings_2 = 512
embedding_dim = 64
seq_len = 512  # Added hyperparameter to control input sequence length
batch_size = 2048  # Added batch size to increase input规模


def get_inputs():
    x1 = torch.randint(0, num_embeddings_1, (batch_size, seq_len))  # Use batch_size and seq_len
    x2 = torch.randint(0, num_embeddings_2, (batch_size, seq_len))  # Use batch_size and seq_len
    return [x1, x2]


def get_init_inputs():
    return [num_embeddings_1, num_embeddings_2, embedding_dim]