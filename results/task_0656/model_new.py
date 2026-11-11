import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, normalized_shape, eps):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.layer_norm_forward(x, self.layer_norm.normalized_shape[0])
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield "step_001", x
        x = cuda_extension.layer_norm_forward(x, self.layer_norm.normalized_shape[0])
        yield "step_002", x
        yield "step_003", x
