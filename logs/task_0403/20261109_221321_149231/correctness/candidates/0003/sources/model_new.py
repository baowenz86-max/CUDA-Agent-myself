import torch
import torch.nn as nn
import cuda_extension


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Use a submodule to reproduce the reference state_dict key 'embedding.weight'
        self.embedding = EmbeddingModule(num_embeddings, embedding_dim)

    def forward(self, x):
        # Step 1: Embedding lookup using custom op
        emb = cuda_extension.embedding_lookup(x, self.embedding.weight)
        # Step 2: L2 normalize using custom op
        norm = cuda_extension.l2_normalize(emb, 1e-9)
        return norm

    def debug_forward(self, *inputs):
        x = inputs[0]
        emb = cuda_extension.embedding_lookup(x, self.embedding.weight)
        yield 'step_001', emb
        norm = cuda_extension.l2_normalize(emb, 1e-9)
        yield 'step_002', norm
        yield 'step_003', norm
