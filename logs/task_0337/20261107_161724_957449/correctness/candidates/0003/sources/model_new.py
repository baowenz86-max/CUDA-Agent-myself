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
        # Generate grid_x and grid_y without torch.meshgrid/torch.arange
        row_indices = torch.tensor(list(range(self.height)), dtype=torch.int64)
        col_indices = torch.tensor(list(range(self.width)), dtype=torch.int64)
        grid_x = row_indices[:, None].expand(-1, self.width)
        grid_y = col_indices[None, :].expand(self.height, -1)
        return x, y, grid_x, grid_y

    def debug_forward(self, *inputs):
        indices, other_indices = inputs
        x = cuda_extension.embedding_lookup(self.embedding.weight, indices)
        yield 'step_001', x
        y = cuda_extension.embedding_lookup(self.embedding.weight, other_indices)
        yield 'step_002', y
        # Generate grid_x and grid_y without torch.meshgrid/torch.arange
        row_indices = torch.tensor(list(range(self.height)), dtype=torch.int64)
        col_indices = torch.tensor(list(range(self.width)), dtype=torch.int64)
        grid_x = row_indices[:, None].expand(-1, self.width)
        grid_y = col_indices[None, :].expand(self.height, -1)
        yield 'step_003', (grid_x, grid_y)
        result = (x, y, grid_x, grid_y)
        yield 'step_004', result
