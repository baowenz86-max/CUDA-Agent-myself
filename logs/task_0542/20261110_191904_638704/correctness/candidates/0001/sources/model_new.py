import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, seq_len, input_dim):
        super().__init__()
        self.embedding_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, indices, batch1, batch2):
        x = cuda_extension.custom_embedding(indices, self.embedding_weight)
        x = cuda_extension.custom_baddbmm(x, batch1, batch2)
        return x

    def debug_forward(self, indices, batch1, batch2):
        x = cuda_extension.custom_embedding(indices, self.embedding_weight)
        yield 'step_001', x
        x = cuda_extension.custom_baddbmm(x, batch1, batch2)
        yield 'step_002', x
        yield 'step_003', x
