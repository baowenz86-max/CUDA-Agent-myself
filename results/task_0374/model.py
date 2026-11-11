import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, bucketize_values):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.bucketize_values = nn.Parameter(torch.tensor(bucketize_values, dtype=torch.float32), requires_grad=False)

    def forward(self, indices, input_values):
        embedded = self.embedding(indices)
        bucketized = torch.bucketize(input_values, self.bucketize_values)
        return embedded, bucketized


num_embeddings = 2048
embedding_dim = 512
bucketize_values = torch.linspace(0.0, 1.0, 1000).tolist()
batch_size = 128
seq_length = 2048


def get_inputs():
    indices = torch.randint(0, num_embeddings, (batch_size, seq_length))
    input_values = torch.rand(batch_size, seq_length)
    return [indices, input_values]


def get_init_inputs():
    return [num_embeddings, embedding_dim, bucketize_values]