import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        emb = self.embed(x)
        amax = torch.amax(emb, dim=1, keepdim=True)
        return amax


num_embeddings = 10000
embedding_dim = 300
batch_size = 512
seq_len = 1024  # Added sequence length to increase input size


def get_inputs():
    return [torch.randint(0, num_embeddings, (batch_size, seq_len))]  # Input shape (512, 1024)


def get_init_inputs():
    return [num_embeddings, embedding_dim]