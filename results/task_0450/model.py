import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        x = self.embedding(x)
        x = self.classifier(x)
        if target is None:
            return x
        loss = self.loss(x, target)
        return x, loss


num_embeddings = 1000
embedding_dim = 300
num_classes = 10
batch_size = 32
seq_len = 8192  # Increased from 512 to increase computation


def get_inputs():
    return [torch.randint(high=num_embeddings, size=(batch_size, seq_len))]


def get_init_inputs():
    return [num_embeddings, embedding_dim, num_classes]