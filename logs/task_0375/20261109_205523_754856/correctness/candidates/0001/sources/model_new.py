import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        # Move input and weight to CUDA for custom ops
        weight_cuda = self.embedding.weight.cuda()
        x_cuda = x.cuda()
        x = cuda_extension.embedding_forward(weight_cuda, x_cuda)
        x = cuda_extension.argsort_forward(x, dim=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        weight_cuda = self.embedding.weight.cuda()
        x_cuda = x.cuda()
        x = cuda_extension.embedding_forward(weight_cuda, x_cuda)
        yield "step_001", x
        x = cuda_extension.argsort_forward(x, dim=1)
        yield "step_002", x
        yield "step_003", x
