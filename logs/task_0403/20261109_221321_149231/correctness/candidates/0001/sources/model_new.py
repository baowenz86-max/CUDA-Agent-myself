import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Use nn.Parameter to store embedding weights for state_dict compatibility
        self.embedding_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        # Initialize randomly (same as reference nn.Embedding default initialization)
        nn.init.normal_(self.embedding_weight)

    def forward(self, x):
        # Step 1: Embedding lookup using custom op
        emb = cuda_extension.embedding_lookup(x, self.embedding_weight)
        # Step 2: L2 normalize using custom op
        norm = cuda_extension.l2_normalize(emb, 1e-9)
        return norm

    def debug_forward(self, *inputs):
        x = inputs[0]
        emb = cuda_extension.embedding_lookup(x, self.embedding_weight)
        yield 'step_001', emb
        norm = cuda_extension.l2_normalize(emb, 1e-9)
        yield 'step_002', norm
        yield 'step_003', norm
