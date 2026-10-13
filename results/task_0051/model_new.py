import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.padding_size = padding_size

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size[0], self.padding_size[1])
        return cuda_extension.sub_scalar_forward(x, 0.5)

    def debug_forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding.weight)
        yield 'step_001', x
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size[0], self.padding_size[1])
        yield 'step_002', x
        result = cuda_extension.sub_scalar_forward(x, 0.5)
        yield 'step_003', result
