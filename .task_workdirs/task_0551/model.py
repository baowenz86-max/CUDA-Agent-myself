import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        return torch.div(x, 2.0)


num_embeddings = 1000
embedding_dim = 512
seq_length = 512  # Hyperparameter to control input sequence length
batch_size = 1024  # Added batch size to increase computation


def get_inputs():
    indices = torch.randint(0, num_embeddings, (batch_size, seq_length))  # Shape (batch_size, seq_length)
    return [indices]


def get_init_inputs():
    return [num_embeddings, embedding_dim]