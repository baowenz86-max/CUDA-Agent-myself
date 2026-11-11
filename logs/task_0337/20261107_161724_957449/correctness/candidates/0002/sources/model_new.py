import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, height, width):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.height = height
        self.width = width

    def forward(self, indices, other_indices):
        x = cuda_extension.embedding_lookup(self.embedding.weight, indices)
        y = cuda_extension.embedding_lookup(self.embedding.weight, other_indices)
        grid_x = torch.tensor([list(range(self.height)) for _ in range(self.width)]).t().contiguous()
        grid_y = torch.tensor([list(range(self.width)) for _ in range(self.height)])
        return x, y, grid_x, grid_y

    def debug_forward(self, *inputs):
        indices, other_indices = inputs
        x = cuda_extension.embedding_lookup(self.embedding.weight, indices)
        yield 'step_001', x
        y = cuda_extension.embedding_lookup(self.embedding.weight, other_indices)
        yield 'step_002', y
        grid_x = torch.tensor([list(range(self.height)) for _ in range(self.width)]).t().contiguous()
        grid_y = torch.tensor([list(range(self.width)) for _ in range(self.height)])
        yield 'step_003', (grid_x, grid_y)
        result = (x, y, grid_x, grid_y)
        yield 'step_004', result
