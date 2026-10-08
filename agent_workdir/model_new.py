import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, bias, batch_first, stack_dim):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, bias=bias, batch_first=batch_first
        )
        self.num_heads = num_heads
        self.stack_dim = stack_dim

    def forward(self, x1, x2, x3, key_padding_mask):
        return cuda_extension.mha_stack_forward(
            x1,
            x2,
            x3,
            key_padding_mask,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads,
            self.stack_dim,
        )
