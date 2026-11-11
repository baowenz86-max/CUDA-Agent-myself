import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        x = cuda_extension.embedding_lookup_forward(x, self.embedding.weight)
        x = cuda_extension.linear_forward(x, self.classifier.weight, self.classifier.bias)
        if target is None:
            return x
        loss = cuda_extension.cross_entropy_loss_forward(x, target)
        return x, loss

    def debug_forward(self, *inputs):
        x = inputs[0]
        target = inputs[1] if len(inputs) > 1 else None
        x_embed = self.embedding(x)
        x_linear = self.classifier(x_embed)
        yield ('step_001', x_embed)
        yield ('step_002', x_linear)
        if target is None:
            yield ('step_003_final', x_linear)
        else:
            loss = self.loss(x_linear, target)
            yield ('step_003_loss', loss)
            yield ('step_004_final', (x_linear, loss))
