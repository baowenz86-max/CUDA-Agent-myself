import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        return cuda_extension.embedding_lookup_forward(
            x, self.embedding.weight, self.embedding.padding_idx
        )

    def debug_forward(self, *inputs):
        x = inputs[0] if inputs else self._debug_x
        out = self.forward(x)
        yield 'step_001', out
