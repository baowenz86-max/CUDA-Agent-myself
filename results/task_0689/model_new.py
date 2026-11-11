import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x1, x2):
        mean_x1 = cuda_extension.mean_dim_forward(x1, self.embedding_dim)
        mean_x2 = cuda_extension.mean_dim_forward(x2, self.embedding_dim)
        return cuda_extension.cosine_similarity_forward(mean_x1, mean_x2)

    def debug_forward(self, x1, x2):
        mean_x1 = cuda_extension.mean_dim_forward(x1, self.embedding_dim)
        yield 'step_001', mean_x1
        mean_x2 = cuda_extension.mean_dim_forward(x2, self.embedding_dim)
        yield 'step_002', mean_x2
        output = cuda_extension.cosine_similarity_forward(mean_x1, mean_x2)
        yield 'step_003', output
