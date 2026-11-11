import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, reduction):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, input, target):
        x = self.embedding(input)
        x = x.view(x.size(0), x.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        x = F.log_softmax(x, dim=-1)
        target = F.softmax(target, dim=-1)
        loss = self.criterion(x, target)
        return loss


batch_size = 512  # Increased to ensure input has a dimension >=512
seq_length = 1000  # Increased sequence length to increase computation
num_embeddings = 5000
embedding_dim = 256
reduction = "batchmean"


def get_inputs():
    input = torch.randint(0, num_embeddings, (batch_size, seq_length))
    target = torch.randn(batch_size, seq_length, embedding_dim)
    return [input, target]


def get_init_inputs():
    return [num_embeddings, embedding_dim, reduction]