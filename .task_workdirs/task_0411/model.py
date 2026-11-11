import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.input_size = input_size

    def forward(self, x, target):
        x = self.embedding(x)
        x = torch.log_softmax(x, dim=-1)
        target = F.softmax(target, dim=-1)
        x = F.kl_div(x, target, reduction="batchmean")
        return x


batch_size = 128
num_embeddings = 1000
embedding_dim = 512
input_size = 10
sequence_length = 1024


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, sequence_length))
    target = torch.randn(batch_size, sequence_length, embedding_dim)
    return [x, target]


def get_init_inputs():
    return [num_embeddings, embedding_dim, input_size]