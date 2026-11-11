import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_embeddings_1, num_embeddings_2, embedding_dim):
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings_1, embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings_2, embedding_dim)

    def forward(self, x1, x2):
        # Move inputs to CUDA if necessary; custom op expects CUDA int64 tensors.
        if not x1.is_cuda:
            x1 = x1.cuda()
        if not x2.is_cuda:
            x2 = x2.cuda()
        # Ensure weights are on the same device as inputs (they should be, but we guard).
        weight1 = self.embedding1.weight.to(x1.device)
        weight2 = self.embedding2.weight.to(x1.device)
        num_embeddings_1 = weight1.shape[0]
        num_embeddings_2 = weight2.shape[0]
        embedding_dim = weight1.shape[1]
        return cuda_extension.embedding_add_forward(
            x1, x2, weight1, weight2,
            num_embeddings_1, num_embeddings_2, embedding_dim
        )

    def debug_forward(self, x1, x2):
        # Same computation as forward but yielding checkpoints.
        if not x1.is_cuda:
            x1 = x1.cuda()
        if not x2.is_cuda:
            x2 = x2.cuda()
        weight1 = self.embedding1.weight.to(x1.device)
        weight2 = self.embedding2.weight.to(x1.device)
        num_embeddings_1 = weight1.shape[0]
        num_embeddings_2 = weight2.shape[0]
        embedding_dim = weight1.shape[1]
        result = cuda_extension.embedding_add_forward(
            x1, x2, weight1, weight2,
            num_embeddings_1, num_embeddings_2, embedding_dim
        )
        yield 'step_001', result
