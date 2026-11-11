import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mode: str = "sum"):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode=mode)

    def forward(self, input, offsets):
        return cuda_extension.embedding_bag_forward(input, offsets, self.embedding_bag.weight)

    def debug_forward(self, input, offsets):
        result = cuda_extension.embedding_bag_forward(input, offsets, self.embedding_bag.weight)
        yield "step_001", result
