import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, loss_target_shape):
        super().__init__()
        # 保持与参考模型相同的参数结构和 state_dict 键
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.loss_target = nn.Parameter(torch.randn(loss_target_shape))

    def forward(self, query, key, value):
        # 确保输入在 CUDA 上且为 float32
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        in_proj_weight = self.multihead_attn.in_proj_weight.contiguous()
        in_proj_bias = self.multihead_attn.in_proj_bias.contiguous()
        out_proj_weight = self.multihead_attn.out_proj.weight.contiguous()
        out_proj_bias = self.multihead_attn.out_proj.bias.contiguous()

        import cuda_extension
        attn_output, attn_output_weights = cuda_extension.multihead_attention_forward(
            query, key, value,
            in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias,
            self.multihead_attn.num_heads,
            self.multihead_attn.batch_first
        )

        cat_output = cuda_extension.concat_dim2(attn_output, query)

        loss_target = self.loss_target.contiguous()
        loss = cuda_extension.smooth_l1_loss(cat_output, loss_target, 1.0)

        return cat_output, attn_output_weights, loss

    def debug_forward(self, query, key, value):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        in_proj_weight = self.multihead_attn.in_proj_weight.contiguous()
        in_proj_bias = self.multihead_attn.in_proj_bias.contiguous()
        out_proj_weight = self.multihead_attn.out_proj.weight.contiguous()
        out_proj_bias = self.multihead_attn.out_proj.bias.contiguous()

        import cuda_extension
        attn_output, attn_output_weights = cuda_extension.multihead_attention_forward(
            query, key, value,
            in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias,
            self.multihead_attn.num_heads,
            self.multihead_attn.batch_first
        )
        yield "step_001", (attn_output, attn_output_weights)

        cat_output = cuda_extension.concat_dim2(attn_output, query)
        yield "step_002", cat_output

        loss_target = self.loss_target.contiguous()
        loss = cuda_extension.smooth_l1_loss(cat_output, loss_target, 1.0)
        yield "step_003", loss

        result = (cat_output, attn_output_weights, loss)
        yield "step_004", result
