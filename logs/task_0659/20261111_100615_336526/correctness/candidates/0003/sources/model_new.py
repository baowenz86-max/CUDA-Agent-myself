import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_size, seq_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_length = seq_length
        # 保持与参考相同的参数结构和名称，确保 state_dict 兼容
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        q = cuda_extension.linear_forward(x1, self.query_linear.weight, self.query_linear.bias)
        k = cuda_extension.linear_forward(x2, self.key_linear.weight, self.key_linear.bias)
        v = cuda_extension.linear_forward(x2, self.value_linear.weight, self.value_linear.bias)
        attn = cuda_extension.scaled_dot_product_attention_forward(q, k, v)
        cat = cuda_extension.concat_forward(x1, attn)
        return cat

    def debug_forward(self, x1, x2):
        q = cuda_extension.linear_forward(x1, self.query_linear.weight, self.query_linear.bias)
        yield 'step_001', q
        k = cuda_extension.linear_forward(x2, self.key_linear.weight, self.key_linear.bias)
        yield 'step_002', k
        v = cuda_extension.linear_forward(x2, self.value_linear.weight, self.value_linear.bias)
        yield 'step_003', v
        attn = cuda_extension.scaled_dot_product_attention_forward(q, k, v)
        yield 'step_004', attn
        cat = cuda_extension.concat_forward(x1, attn)
        yield 'step_005', cat
        yield 'step_006', cat
