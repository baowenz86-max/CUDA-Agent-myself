import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # 使用 register_parameter 以便 state_dict 键为 "embedding.weight"，与参考模型一致
        self.embedding_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.embedding_weight)
        self.register_parameter('embedding.weight', self.embedding_weight)

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
