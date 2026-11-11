import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, target):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Assuming a simple mean for sequence embedding
        logits = self.fc(x)
        loss = nn.BCEWithLogitsLoss()(logits, target)
        return logits, loss


batch_size = 512  # Increased to 512 to make input dimension >=512
num_embeddings = 1000
embedding_dim = 128
num_classes = 1
seq_length = 2000  # Increased sequence length to increase computation


def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, seq_length))  # Shape becomes (512, 2000)
    target = torch.randint(0, 2, (batch_size, num_classes)).float()  # Shape (512, 1)
    return [x, target]


def get_init_inputs():
    return [num_embeddings, embedding_dim, num_classes]