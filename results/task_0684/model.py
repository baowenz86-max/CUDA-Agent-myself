import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        # x should be a 1D tensor of long or int dtype
        embedded = self.embedding(x)
        # Split along the first dimension (dim=0) into 2 equal sized tensors
        split_tensors = torch.split(embedded, embedded.size(dim=0) // 2, dim=0)
        return split_tensors


# Define the number of embeddings, their dimensions, and the indices for splitting
num_embeddings = 1000
embedding_dim = 512
indices = torch.tensor([i for i in range(num_embeddings)])


def get_inputs():
    # Randomly sample indices from the embedding space with increased input size
    indices = torch.randint(low=0, high=num_embeddings, size=(1000000,))
    return [indices]


def get_init_inputs():
    return [num_embeddings, embedding_dim]