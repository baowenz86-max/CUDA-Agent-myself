import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, repeat_interleave_times, scaling_factor):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if repeat_interleave_times < 0:
            raise ValueError("repeat_interleave_times must be nonnegative")
        self.multihead_attn = nn.Module()
        self.multihead_attn.embed_dim = embed_dim
        self.multihead_attn.num_heads = num_heads
        self.multihead_attn.batch_first = batch_first
        self.multihead_attn.in_proj_weight = nn.Parameter(torch.randn(3 * embed_dim, embed_dim))
        self.multihead_attn.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.multihead_attn.out_proj = nn.Module()
        self.multihead_attn.out_proj.weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.multihead_attn.out_proj.bias = nn.Parameter(torch.zeros(embed_dim))
        self.repeat_interleave_times = repeat_interleave_times
        self.scaling_factor = scaling_factor

    def forward(self, x, y):
        if len(x.shape) != 3 or len(y.shape) != 3:
            raise ValueError("x and y must be three-dimensional")
        attention = self.multihead_attn
        batch_axis = 0 if attention.batch_first else 1
        sequence_axis = 1 if attention.batch_first else 0
        batch = x.shape[batch_axis]
        queries = x.shape[sequence_axis]
        keys = y.shape[sequence_axis]
        embed = attention.embed_dim
        heads = attention.num_heads
        depth = embed // heads
        repeats = self.repeat_interleave_times
        if y.shape[batch_axis] != batch or x.shape[2] != embed or y.shape[2] != embed:
            raise ValueError("input batch and embedding dimensions must match")
        output_shape = (batch, queries * repeats, embed) if attention.batch_first else (queries, batch * repeats, embed)
        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        if batch == 0 or queries == 0 or repeats == 0:
            return cuda_extension.fused_attention_forward(
                x, y, attention.in_proj_weight, attention.in_proj_bias,
                attention.out_proj.weight, attention.out_proj.bias,
                output, output, output, output, output, output,
                output, output, output, output, output,
                heads, attention.batch_first, repeats, self.scaling_factor,
            )
        query_projection = torch.empty((batch * queries, embed), device=x.device, dtype=x.dtype)
        key_projection = torch.empty((batch * keys, embed), device=x.device, dtype=x.dtype)
        value_projection = torch.empty((batch * keys, embed), device=x.device, dtype=x.dtype)
        query = torch.empty((batch, heads, queries, depth), device=x.device, dtype=x.dtype)
        key = torch.empty((batch, heads, keys, depth), device=x.device, dtype=x.dtype)
        value = torch.empty((batch, heads, keys, depth), device=x.device, dtype=x.dtype)
        probabilities = torch.empty((batch, heads, queries, keys), device=x.device, dtype=x.dtype)
        context = torch.empty((batch, heads, queries, depth), device=x.device, dtype=x.dtype)
        merged = torch.empty((batch * queries, embed), device=x.device, dtype=x.dtype)
        projected = torch.empty((batch * queries, embed), device=x.device, dtype=x.dtype)
        return cuda_extension.fused_attention_forward(
            x, y, attention.in_proj_weight, attention.in_proj_bias,
            attention.out_proj.weight, attention.out_proj.bias,
            query_projection, key_projection, value_projection,
            query, key, value, probabilities, context, merged, projected, output,
            heads, attention.batch_first, repeats, self.scaling_factor,
        )
