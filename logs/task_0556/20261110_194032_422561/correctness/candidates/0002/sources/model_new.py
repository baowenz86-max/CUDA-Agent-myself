import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, vocab_size, embedding_dim, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # Keep CPU parameters as state_dict; custom ops will move them as needed

    def forward(self, x):
        # x may be CPU or CUDA depending on harness; custom_embedding handles both
        x = cuda_extension.custom_embedding(x, self.embedding.weight, self.embedding.num_embeddings, self.embedding.embedding_dim)
        x = cuda_extension.custom_layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps, self.norm.normalized_shape[0], self.norm.elementwise_affine)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_embedding(x, self.embedding.weight, self.embedding.num_embeddings, self.embedding.embedding_dim)
        yield ('step_001', x)
        x = cuda_extension.custom_layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps, self.norm.normalized_shape[0], self.norm.elementwise_affine)
        yield ('step_002', x)
        yield ('step_003', x)
