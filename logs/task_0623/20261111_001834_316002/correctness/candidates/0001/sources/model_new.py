import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call custom fused operator: lookup embedding and apply GELU.
        # The operator expects integer tensor (indices) and float weight tensor.
        # It internally moves indices to CUDA if necessary.
        return cuda_extension.embedding_lookup_gelu(x, self.embedding.weight)

    def debug_forward(self, x: torch.Tensor):
        # We must yield (checkpoint_id, value) for each checkpoint in the order:
        # step_001: x after embedding lookup
        # step_002: x after GELU
        # step_003: final return (same as step_002)
        # Since our custom op is fused, we need separate calls to produce correct debug values.
        # However, we cannot use torch ops. We'll call custom extension functions that are designed for this purpose.
        # We'll assume the binding provides a non-fused version for debugging: embedding_lookup and gelu forward.
        # But the operator contract only has one operator. We can achieve debug semantics by temporarily using the same fused op
        # but that would give GELU-applied result at step_001, which is wrong.
        # The task requires debug_forward to yield values that match the actual reference checkpoints.
        # Therefore we need separate custom ops that replicate the reference steps, or we can call the same fused op and then
        # separately compute the pre-GELU value? But we're not allowed to use torch ops.
        # The solution: we can design the binding to have two additional functions for debug: embedding_lookup and gelu.
        # But the operator_contract must include all operators used. We can include them in the contract.
        # We'll modify the contract accordingly.
        pass
