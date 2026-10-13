import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, loss_target_shape):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.loss_target = nn.Parameter(torch.randn(loss_target_shape))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, query, key, value):
        # Ensure all inputs are on CUDA
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        in_proj_weight = self.multihead_attn.in_proj_weight.cuda()
        in_proj_bias = self.multihead_attn.in_proj_bias.cuda()
        out_proj_weight = self.multihead_attn.out_proj.weight.cuda()
        out_proj_bias = self.multihead_attn.out_proj.bias.cuda()

        attn_output, attn_output_weights = cuda_extension.mha_forward(
            query, key, value,
            in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias,
            self.num_heads, self.batch_first
        )

        cat_output = cuda_extension.concat_dim2(attn_output, query)

        loss = cuda_extension.smooth_l1_loss(cat_output, self.loss_target.cuda())

        return cat_output, attn_output_weights, loss

    def debug_forward(self, query, key, value):
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        in_proj_weight = self.multihead_attn.in_proj_weight.cuda()
        in_proj_bias = self.multihead_attn.in_proj_bias.cuda()
        out_proj_weight = self.multihead_attn.out_proj.weight.cuda()
        out_proj_bias = self.multihead_attn.out_proj.bias.cuda()

        attn_output, attn_output_weights = cuda_extension.mha_forward(
            query, key, value,
            in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias,
            self.num_heads, self.batch_first
        )
        yield ('step_001', (attn_output, attn_output_weights))

        cat_output = cuda_extension.concat_dim2(attn_output, query)
        yield ('step_002', cat_output)

        loss = cuda_extension.smooth_l1_loss(cat_output, self.loss_target.cuda())
        yield ('step_003', loss)

        yield ('step_004', (cat_output, attn_output_weights, loss))
