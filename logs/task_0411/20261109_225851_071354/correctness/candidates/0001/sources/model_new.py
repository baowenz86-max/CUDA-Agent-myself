import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.input_size = input_size

    def forward(self, x, target):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.log_softmax_forward(x)
        target = cuda_extension.softmax_forward(target)
        x = cuda_extension.kl_div_batchmean_forward(x, target)
        return x

    def debug_forward(self, x, target):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield 'step_001', x
        x = cuda_extension.log_softmax_forward(x)
        yield 'step_002', x
        target = cuda_extension.softmax_forward(target)
        yield 'step_003', target
        x = cuda_extension.kl_div_batchmean_forward(x, target)
        yield 'step_004', x
        yield 'step_005', x
