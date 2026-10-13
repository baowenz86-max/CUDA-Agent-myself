import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, loss_target_shape):
        super().__init__()
        # 保持与参考模型相同的子模块和参数，确保 state_dict 键一致
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.loss_target = nn.Parameter(torch.randn(loss_target_shape))

    def forward(self, query, key, value):
        # 步骤 1：与参考 self.multihead_attn(query, key, value) 对应
        attn_output, attn_output_weights = cuda_extension.multihead_attn_forward(
            query, key, value,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.multihead_attn.num_heads,
            self.multihead_attn.batch_first
        )

        # 步骤 2：与参考 torch.cat([attn_output, query], dim=2) 对应
        cat_output = cuda_extension.cat_forward(attn_output, query, dim=2)

        # 步骤 3：与参考 F.smooth_l1_loss(cat_output, self.loss_target) 对应
        loss = cuda_extension.smooth_l1_loss_forward(cat_output, self.loss_target)

        # 最终返回结构与参考一致
        return cat_output, attn_output_weights, loss

    def debug_forward(self, query, key, value):
        # 与 forward 相同的计算流程，但按参考 checkpoint 顺序 yield (id, value)
        # 注意：attn_output_weights 在自定义算子返回中应保持为 torch.Tensor，
        # multihead_attn_forward 返回的 tuple 结构与参考一致。
        attn_result = cuda_extension.multihead_attn_forward(
            query, key, value,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.multihead_attn.num_heads,
            self.multihead_attn.batch_first
        )
        attn_output, attn_output_weights = attn_result
        yield ('step_001', attn_result)  # 保持与 checkpoints 中 value 结构一致的 tuple

        cat_output = cuda_extension.cat_forward(attn_output, query, dim=2)
        yield ('step_002', cat_output)

        loss = cuda_extension.smooth_l1_loss_forward(cat_output, self.loss_target)
        yield ('step_003', loss)

        final = (cat_output, attn_output_weights, loss)
        yield ('step_004', final)
