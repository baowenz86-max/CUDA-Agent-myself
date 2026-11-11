import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        weight = self.embedding.weight
        _, final_out = cuda_extension.embedding_log_softmax_forward(x, weight, 1)
        return final_out

    def debug_forward(self, x):
        weight = self.embedding.weight
        embedding_out, final_out = cuda_extension.embedding_log_softmax_forward(x, weight, 1)
        yield 'step_001', embedding_out
        yield 'step_002', final_out
        yield 'step_003', final_out
