import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids, dim: int = 0, index: torch.Tensor = None):
        embedded = cuda_extension.embedding_lookup_forward(
            self.embedding.weight, input_ids
        )
        index_selected = cuda_extension.index_select_forward(
            embedded, dim, index
        )
        return index_selected

    def debug_forward(self, input_ids, dim: int = 0, index: torch.Tensor = None):
        embedded = cuda_extension.embedding_lookup_forward(
            self.embedding.weight, input_ids
        )
        yield 'step_001', embedded
        index_selected = cuda_extension.index_select_forward(
            embedded, dim, index
        )
        yield 'step_002', index_selected
        yield 'step_003', index_selected
