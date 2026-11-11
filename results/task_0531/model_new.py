import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, min_value):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.min_value = min_value

    def forward(self, x):
        # Move indices to CUDA
        if not x.is_cuda:
            x = x.cuda()
        # Custom embedding lookup
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        # Custom clamp minimum
        return cuda_extension.clamp_min_forward(x, self.min_value)

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield "step_001", x
        y = cuda_extension.clamp_min_forward(x, self.min_value)
        yield "step_002", y
