import torch
import torch.nn as nn
import cuda_extension


class _LinearParams(nn.Module):
    """Container for linear parameters to match state_dict keys."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        # Initialize as nn.Linear would (but we don't call torch ops here? 
        # We can use nn.init functions, which are allowed? They are torch operations on parameters.
        # SKILL.md says only tensor creation and custom ops allowed in model_new.py, but initialization is not forward; 
        # However, to be safe, we can initialize with torch.empty and later load_state_dict will overwrite. 
        # The reference fixture provides state_dict, so initialization values don't matter for correctness.
        # But for completeness, we call reset_parameters using allowed torch.nn.init? 
        # nn.init.kaiming_uniform_ is a tensor method; it's not in forward, so likely allowed, but to avoid risk, we just leave empty.
        # Verification loads state_dict before forward, so parameters get correct values.
        # We'll not call any init to avoid any issues.

    def forward(self, x):
        raise NotImplementedError("Should not be called; use custom op")


class _BCEWithLogitsLossPlaceholder(nn.Module):
    """Placeholder to match module list; no parameters."""
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        raise NotImplementedError("Should not be called; use custom op")


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        # Use custom parameter-only modules to keep state_dict keys W_q.weight, W_q.bias, etc.
        self.W_q = _LinearParams(embed_dim, embed_dim, bias=bias)
        self.W_k = _LinearParams(embed_dim, embed_dim, bias=bias)
        self.W_v = _LinearParams(embed_dim, embed_dim, bias=bias)
        self.bce_loss = _BCEWithLogitsLossPlaceholder()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        # Step 1: q = self.W_q(query) + self.pos_enc[:, : query.shape[1], :]
        q_linear = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        pos_q = self.pos_enc[:, : query.shape[1], :]
        q = cuda_extension.add_pos_enc(q_linear, pos_q)

        # Step 2: k = self.W_k(key) + self.pos_enc[:, : key.shape[1], :]
        k_linear = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        pos_k = self.pos_enc[:, : key.shape[1], :]
        k = cuda_extension.add_pos_enc(k_linear, pos_k)

        # Step 3: v = self.W_v(value) + self.pos_enc[:, : value.shape[1], :]
        v_linear = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)
        pos_v = self.pos_enc[:, : value.shape[1], :]
        v = cuda_extension.add_pos_enc(v_linear, pos_v)

        # Step 4: attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_output = cuda_extension.sdpa_forward(q, k, v, attn_mask)

        # Step 5: logits = torch.baddbmm(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))
        zeros = torch.zeros_like(target)
        attn_output_T = attn_output.transpose(-2, -1).contiguous()  # ensure contiguous for custom op
        logits = cuda_extension.baddbmm_forward(zeros, attn_output, attn_output_T)

        # Step 6: loss = self.bce_loss(logits, target)
        loss = cuda_extension.bce_loss_forward(logits, target)

        return loss

    def debug_forward(self, query, key, value, attn_mask, target):
        # Replicate forward step by step, yielding checkpoints exactly as reference trace.
        # Note: Must not change computation semantics; use same custom ops.
        # Step 1
        q_linear = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        pos_q = self.pos_enc[:, : query.shape[1], :]
        q = cuda_extension.add_pos_enc(q_linear, pos_q)
        yield ("step_001", q)

        # Step 2
        k_linear = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        pos_k = self.pos_enc[:, : key.shape[1], :]
        k = cuda_extension.add_pos_enc(k_linear, pos_k)
        yield ("step_002", k)

        # Step 3
        v_linear = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)
        pos_v = self.pos_enc[:, : value.shape[1], :]
        v = cuda_extension.add_pos_enc(v_linear, pos_v)
        yield ("step_003", v)

        # Step 4
        attn_output = cuda_extension.sdpa_forward(q, k, v, attn_mask)
        yield ("step_004", attn_output)

        # Step 5
        zeros = torch.zeros_like(target)
        attn_output_T = attn_output.transpose(-2, -1).contiguous()
        logits = cuda_extension.baddbmm_forward(zeros, attn_output, attn_output_T)
        yield ("step_005", logits)

        # Step 6
        loss = cuda_extension.bce_loss_forward(logits, target)
        yield ("step_006", loss)

        # Step 7: return loss
        yield ("step_007", loss)
        return  # debug generator should not return a value, just yield
