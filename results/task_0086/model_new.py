import torch
import torch.nn as nn
from torch import Tensor

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad, embed_dim, num_heads, bias):
        super().__init__()
        self.pad = tuple(pad)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q_padded = cuda_extension.reflection_pad2d_forward(q.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        k_padded = cuda_extension.reflection_pad2d_forward(k.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        v_padded = cuda_extension.reflection_pad2d_forward(v.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        out = cuda_extension.scaled_dot_product_attention_forward(q_padded, k_padded, v_padded, self.bias)
        return out

    def debug_forward(self, q: Tensor, k: Tensor, v: Tensor):
        q_padded = cuda_extension.reflection_pad2d_forward(q.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        yield 'step_001', q_padded
        k_padded = cuda_extension.reflection_pad2d_forward(k.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        yield 'step_002', k_padded
        v_padded = cuda_extension.reflection_pad2d_forward(v.to('cuda'), self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        yield 'step_003', v_padded
        out = cuda_extension.scaled_dot_product_attention_forward(q_padded, k_padded, v_padded, self.bias)
        yield 'step_004', out
        yield 'step_005', out
