import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        weight = self.embed.weight
        if not weight.is_cuda:
            weight = weight.cuda()
        emb = cuda_extension.embedding_forward(weight, x)
        amax = cuda_extension.amax_dim1_keepdim(emb)
        return amax

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        weight = self.embed.weight
        if not weight.is_cuda:
            weight = weight.cuda()
        emb = cuda_extension.embedding_forward(weight, x)
        yield 'step_001', emb
        amax = cuda_extension.amax_dim1_keepdim(emb)
        yield 'step_002', amax
        yield 'step_003', amax
