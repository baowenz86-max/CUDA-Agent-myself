import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.squeeze(x)
        return x


num_embeddings = 1000
embedding_dim = 300
input_length = 512  # Added hyperparameter to control input length
batch_size = 1024  # Added batch size to increase input规模


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, input_length), dtype=torch.long)]


def get_init_inputs():
    return [num_embeddings, embedding_dim]