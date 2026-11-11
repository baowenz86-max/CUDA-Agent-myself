import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, vocab_size, embedding_dim, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        # Preserve state_dict keys and CPU storage
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # Create CUDA copies for custom kernels (non-parameter, not in state_dict)
        self._embedding_weight_cuda = self.embedding.weight.detach().cuda()
        if elementwise_affine:
            self._norm_weight_cuda = self.norm.weight.detach().cuda()
            self._norm_bias_cuda = self.norm.bias.detach().cuda()
        else:
            self._norm_weight_cuda = None
            self._norm_bias_cuda = None

    def forward(self, x):
        # Custom embedding: indices on CPU, returns CUDA tensor
        x = cuda_extension.custom_embedding(x, self.embedding.weight, self.embedding.num_embeddings, self.embedding.embedding_dim)
        # Custom layer norm: input CUDA, weight/bias CPU (or None), eps, normalized_dim, elementwise_affine
        x = cuda_extension.custom_layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps, self.norm.normalized_shape[0], self.norm.elementwise_affine)
        return x

    def debug_forward(self, x):
        # Each checkpoint must match reference values and order
        x = cuda_extension.custom_embedding(x, self.embedding.weight, self.embedding.num_embeddings, self.embedding.embedding_dim)
        yield ('step_001', x)
        x = cuda_extension.custom_layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps, self.norm.normalized_shape[0], self.norm.elementwise_affine)
        yield ('step_002', x)
        yield ('step_003', x)
