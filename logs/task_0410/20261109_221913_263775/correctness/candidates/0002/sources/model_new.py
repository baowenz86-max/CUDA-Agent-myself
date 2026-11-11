import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_size):
        super().__init__()
        # 使用子模块 embedding 来匹配参考 state_dict 键 'embedding.weight'
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embedding_weight = self.embedding.weight  # 便于 forward 引用
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.output_size = output_size

    def forward(self, x):
        x = cuda_extension.embedding_forward(x, self.embedding_weight, self.num_embeddings, self.embedding_dim)
        x = cuda_extension.adaptive_avg_pool2d_1x1(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.embedding_forward(x, self.embedding_weight, self.num_embeddings, self.embedding_dim)
        yield 'step_001', x
        x = cuda_extension.adaptive_avg_pool2d_1x1(x)
        yield 'step_002', x
        yield 'step_003', x
