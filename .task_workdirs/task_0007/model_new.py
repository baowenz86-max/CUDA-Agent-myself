import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        q_weight = self.W_q.weight
        q_bias = self.W_q.bias
        k_weight = self.W_k.weight
        k_bias = self.W_k.bias
        v_weight = self.W_v.weight
        v_bias = self.W_v.bias

        q = cuda_extension.linear_forward(query, q_weight, q_bias)
        k = cuda_extension.linear_forward(key, k_weight, k_bias)
        v = cuda_extension.linear_forward(value, v_weight, v_bias)

        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)

        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)

        batch2 = attn_output.transpose(-2, -1).contiguous()
        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, batch2)

        loss = cuda_extension.bce_with_logits_loss(logits, target)
        return loss

    def debug_forward(self, query, key, value, attn_mask, target):
        q_weight = self.W_q.weight
        q_bias = self.W_q.bias
        k_weight = self.W_k.weight
        k_bias = self.W_k.bias
        v_weight = self.W_v.weight
        v_bias = self.W_v.bias

        q = cuda_extension.linear_forward(query, q_weight, q_bias)
        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        yield "step_001", q

        k = cuda_extension.linear_forward(key, k_weight, k_bias)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        yield "step_002", k

        v = cuda_extension.linear_forward(value, v_weight, v_bias)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)
        yield "step_003", v

        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)
        yield "step_004", attn_output

        batch2 = attn_output.transpose(-2, -1).contiguous()
        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, batch2)
        yield "step_005", logits

        loss = cuda_extension.bce_with_logits_loss(logits, target)
        yield "step_006", loss

        yield "step_007", loss
