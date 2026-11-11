import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids, dim: int = 0, index: torch.Tensor = None):
        embedded = self.embedding(input_ids)
        index_selected = torch.index_select(embedded, dim, index)
        return index_selected


num_embeddings = 10000
embedding_dim = 1024
batch_size = 64
input_seq_len = 2048


def get_inputs():
    input_ids = torch.randint(0, num_embeddings, (batch_size, input_seq_len))
    dim = 1
    index = torch.randint(0, input_seq_len, (1024,))
    return [input_ids, dim, index]


def get_init_inputs():
    return [num_embeddings, embedding_dim]