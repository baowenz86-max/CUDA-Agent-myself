import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        # Use nn.Linear to match state_dict keys: W_q.weight, W_q.bias, etc.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        # Linear projections using custom op (no torch ops)
        q = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        k = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        v = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)

        # Add positional encoding
        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)

        # Scaled dot product attention
        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)

        # baddbmm with beta=0, alpha=1 on zeros_like(target)
        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))

        # BCE loss
        loss = cuda_extension.bce_with_logits_loss(logits, target)
        return loss

    def debug_forward(self, query, key, value, attn_mask, target):
        # Step-by-step yield matching reference checkpoint IDs
        q = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        yield "step_001", q

        k = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        yield "step_002", k

        v = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)
        yield "step_003", v

        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)
        yield "step_004", attn_output

        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))
        yield "step_005", logits

        loss = cuda_extension.bce_with_logits_loss(logits, target)
        yield "step_006", loss

        yield "step_007", loss
