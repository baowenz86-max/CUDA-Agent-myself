import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, bucketize_values):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.bucketize_values = nn.Parameter(torch.tensor(bucketize_values, dtype=torch.float32), requires_grad=False)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices, input_values):
        embedded = cuda_extension.custom_embedding_layernorm(
            indices, self.embedding[0].weight, self.embedding[1].weight,
            self.embedding[1].bias, 1e-05, self.num_embeddings, self.embedding_dim
        )
        bucketized = cuda_extension.custom_bucketize(input_values, self.bucketize_values)
        return embedded, bucketized

    def debug_forward(self, indices, input_values):
        embedded = cuda_extension.custom_embedding_layernorm(
            indices, self.embedding[0].weight, self.embedding[1].weight,
            self.embedding[1].bias, 1e-05, self.num_embeddings, self.embedding_dim
        )
        yield 'step_001', embedded
        bucketized = cuda_extension.custom_bucketize(input_values, self.bucketize_values)
        yield 'step_002', bucketized
        result = (embedded, bucketized)
        yield 'step_003', result
