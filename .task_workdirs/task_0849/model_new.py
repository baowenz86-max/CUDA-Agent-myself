import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, sequence_length, repeat_interleave_times):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.sequence_length = sequence_length
        self.repeat_interleave_times = repeat_interleave_times

    def forward(self, x):
        # Custom op for MultiheadAttention: returns (output, weights)
        attn_output, _ = torch.ops.cuda_extension.custom_multihead_attention_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.multihead_attn.embed_dim,
            self.multihead_attn.num_heads,
        )
        x = torch.ops.cuda_extension.custom_repeat_interleave_forward(
            attn_output, self.repeat_interleave_times, 1
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        attn_output, attn_weights = torch.ops.cuda_extension.custom_multihead_attention_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.multihead_attn.embed_dim,
            self.multihead_attn.num_heads,
        )
        yield 'step_001', (attn_output, attn_weights)
        x = torch.ops.cuda_extension.custom_repeat_interleave_forward(
            attn_output, self.repeat_interleave_times, 1
        )
        yield 'step_002', x
        yield 'step_003', x
