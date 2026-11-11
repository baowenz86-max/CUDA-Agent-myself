import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = nn.GELU()(x)
        return x


num_embeddings = 20000  # number of unique tokens in the source text
embedding_dim = 768  # dimension of the embeddings
batch_size = 256  # Increased batch size to 256
sequence_length = 512  # Sequence length remains 512


def get_inputs():
    # token ids should be in the range [0, num_embeddings-1]
    x = torch.randint(0, num_embeddings, (batch_size, sequence_length))  # Now shape (256, 512)
    return [x]


def get_init_inputs():
    return [num_embeddings, embedding_dim]