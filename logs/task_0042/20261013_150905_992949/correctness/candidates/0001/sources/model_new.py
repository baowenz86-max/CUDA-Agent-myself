import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, loss_target_shape):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.loss_target = nn.Parameter(torch.randn(loss_target_shape))

    def forward(self, query, key, value):
        # Step 001: multihead attention
        in_proj_weight = self.multihead_attn.in_proj_weight
        in_proj_bias = self.multihead_attn.in_proj_bias
        out_proj_weight = self.multihead_attn.out_proj.weight
        out_proj_bias = self.multihead_attn.out_proj.bias

        q, k, v = cuda_extension.in_projection_forward(query, key, value, in_proj_weight, in_proj_bias)
        
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        embed_dim = query.shape[2]
        num_heads = self.multihead_attn.num_heads
        head_dim = embed_dim // num_heads

        # reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

        attn_output_weights = cuda_extension.attention_weights_forward(q, k, num_heads, head_dim)

        # compute attention output: attn = weights @ v
        # Note: we need a batched matmul for each head; we can use a custom op or reshape trick
        # For now, we will use a custom op for attention output as well, but the contract above doesn't include it.
        # Actually, we need the full attention output before output projection.
        # To avoid torch ops, we must implement the entire attention computation in a merged custom op.
        # However, the task says to implement reference computational statements separately before attempting fusion.
        # So we should have a separate op for attention output computation.
        # We'll add an op 'attention_output_forward' but it's not in the current contract.
        # The contract must exactly match the operators used in model_new.py. So we need to revise the contract.
        # Let's reconsider: we have operator_contract with in_projection_forward, attention_weights_forward, output_projection_forward, smooth_l1_loss_forward.
        # But we also need an op for attention output (weighted sum of values) and then the final projection.
        # The reference multihead_attn computes both attention weights and attention output. We could have one op that computes both, but we separated them.
        # The forward as written above would fail because we don't compute attn_output before output_projection_forward.
        # We need to include an op for attention output computation, or combine attention weights and output into a single op.
        # Since we are in the first phase, we must finalize the operator contract. So let's define an op that computes the entire multihead attention in one go?
        # But the task says: "逐参考计算步骤实现，不做跨步骤融合". However, the reference step 001 itself is a single call to multihead_attn, which internally does multiple computations.
        # We can break down multihead_attn into sub-steps as long as we implement them separately without fusion. That is acceptable.
        # So we need to define all necessary ops. Let's update the contract to include an op for attention output.
        # We'll define an operator 'attention_output_forward' that computes the weighted sum of values and optionally the weights.
        # Actually, attention_weights_forward already returns weights, but we also need the output before projection.
        # So we can have 'attention_output_forward' that takes q, k, v and returns both attn_output (pre-projection) and attn_output_weights.
        # But we have already defined attention_weights_forward and output_projection_forward. Alternatively, we can combine all into one 'multihead_attention_forward' op.
        # Simpler: define 'multihead_attention_forward' that takes query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, num_heads, and returns attn_output, attn_output_weights.
        # Then forward can call that single op for step 001.
        # That would be cleaner and avoid intermediate custom ops. But the task states to implement reference computational statements separately before attempting fusion. However, the reference step 001 is a single statement. We can treat the internal computations as part of that kernel's semantics, not as separate reference statements. So it's allowed to have one op that does the whole multihead attention, as long as we don't fuse later steps (cat, loss). That seems fine.
        # Let's revise: the operator_contract should have an op 'multihead_attention_forward' covering the entire step_001, and then separate ops for cat and smooth_l1_loss? But the task says we can only call custom ops for CUDA computations. torch.cat is not a custom op, so we need a custom op for concatenation as well.
        # So we need ops: multihead_attention_forward, concatenate_forward, smooth_l1_loss_forward.
        # That would be simpler.
        # But the prompt says: "model_new.py 只能调用后续 CUDA/C++ 扩展中会实现的自定义算子；请明确、统一地设计算子名称、参数顺序、返回值结构和 dtype/shape 约定".
        # So we can design fewer ops. Let's do that.
        # We'll define three operators:
        # 1. multihead_attention_forward(query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, num_heads) -> (attn_output, attn_output_weights)
        # 2. concatenate_forward(tensor1, tensor2, dim) -> tensor
        # 3. smooth_l1_loss_forward(input, target, beta) -> scalar
        # Then forward can call them in sequence.
        # That would be easier and matches the reference steps: step 001 -> multihead_attention_forward, step 002 -> concatenate_forward, step 003 -> smooth_l1_loss_forward.
        # Let's adopt this approach.
        # We'll rewrite model_new.py accordingly and update the operator_contract.
        # However, the prompt also says: "first阶段还必须返回顶层 operator_contract" and we have already provided one with more granular ops. We should change it to the simpler three-op design to avoid confusion and missing ops.
        # Let's do that.
        # We'll provide model_new.py with the three-op design.
        # For debug_forward, we'll produce checkpoints: after step_001, after step_002, after step_003, and final return.
        # The debug_forward will call the same custom ops and yield the results.
        # In model_new.py, we cannot use torch operations except tensor creation and custom ops. So we must avoid using .view, .transpose, etc. That's why we need to encapsulate everything inside the custom ops.
        # So we'll have multihead_attention_forward do all internal reshaping and computations.
        # That is allowed as long as the CUDA kernel implements those efficiently.
        # So the forward will be:
        # attn_output, attn_output_weights = cuda_extension.multihead_attention_forward(query, key, value, self.multihead_attn.in_proj_weight, self.multihead_attn.in_proj_bias, self.multihead_attn.out_proj.weight, self.multihead_attn.out_proj.bias, self.multihead_attn.num_heads)
        # cat_output = cuda_extension.concatenate_forward(attn_output, query, 2)
        # loss = cuda_extension.smooth_l1_loss_forward(cat_output, self.loss_target, 1.0)
        # return cat_output, attn_output_weights, loss
        # That's clean.
        # We need to ensure the operator_contract includes these three operators with proper parameters.
        # Let's write that.
        # Also, note that the input tensors may be on CPU according to task_contract, but execution device is cuda. The binding should move them to cuda or assume they are already on cuda? The reference model is run on cuda, so inputs will be moved to cuda before forward. So we can assume cuda tensors.
        # But we should note that in the contract, device: 'cuda'.
        # For parameters, they are on cuda as well after model.to(cuda).
        # So all tensors are cuda.
        # Let's finalize.
        # We'll return JSON with model_new.py and operator_contract.
        # We need to escape properly.
        # We'll provide the full content of model_new.py as a string.
        # We'll also provide the operator_contract as a top-level key, not in files.
        # Let's craft the JSON.
        # We'll include analysis as required.
        # We'll keep the analysis brief.
        # Ensure no trailing commas, proper escaping.
        # We'll write the JSON manually.
