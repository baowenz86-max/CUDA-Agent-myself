import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, height, width, depth):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = depth

        # Use submodules to match reference state_dict keys
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.bn = nn.BatchNorm3d(embedding_dim)

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.embedding_forward(
            x, self.embedding.weight,
            self.num_embeddings, self.embedding_dim
        )
        x = cuda_extension.permute_4d_to_bn3d(
            x, self.batch_size, self.embedding_dim,
            self.height, self.width, self.depth
        )
        x = cuda_extension.bn3d_eval_forward(
            x, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        x = cuda_extension.permute_bn3d_to_4d(
            x, self.batch_size, self.embedding_dim,
            self.height, self.width, self.depth
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = x.cuda()
        yield 'step_001', x

        x = cuda_extension.embedding_forward(
            x, self.embedding.weight,
            self.num_embeddings, self.embedding_dim
        )
        yield 'step_002', x

        x = cuda_extension.permute_4d_to_bn3d(
            x, self.batch_size, self.embedding_dim,
            self.height, self.width, self.depth
        )
        yield 'step_003', x

        x = cuda_extension.bn3d_eval_forward(
            x, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        yield 'step_004', x

        x = cuda_extension.permute_bn3d_to_4d(
            x, self.batch_size, self.embedding_dim,
            self.height, self.width, self.depth
        )
        yield 'step_005', x
