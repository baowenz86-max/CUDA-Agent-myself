import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mode: str = "sum"):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode=mode)

    def forward(self, input, offsets):
        return self.embedding_bag(input, offsets)


num_embeddings = 10
embedding_dim = 5
input_length = 10240  # Hyperparameter to control input tensor length


def get_inputs():
    input = torch.randint(0, num_embeddings, (input_length,))  # Shape: (10240,)
    offsets = torch.tensor([0, input_length])  # Offsets for a single bag
    return [input, offsets]


def get_init_inputs():
    return [num_embeddings, embedding_dim]