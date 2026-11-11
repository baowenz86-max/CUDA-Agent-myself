import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, target):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.mean_dim_forward(x, 1)
        logits = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        loss = cuda_extension.bce_with_logits_forward(logits, target)
        return logits, loss

    def debug_forward(self, x, target):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield 'step_001', x
        x = cuda_extension.mean_dim_forward(x, 1)
        yield 'step_002', x
        logits = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        yield 'step_003', logits
        loss = cuda_extension.bce_with_logits_forward(logits, target)
        yield 'step_004', loss
        yield 'step_005', (logits, loss)
