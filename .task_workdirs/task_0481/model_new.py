import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_out_features, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear_out_features = linear_out_features

        # Attention parameters: in_proj_weight/bias and out_proj.weight/bias
        self.attention_in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.attention_in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.attention_out_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.attention_out_proj_bias = nn.Parameter(torch.empty(embed_dim))

        # Linear parameters
        self.linear_weight = nn.Parameter(torch.empty(linear_out_features, embed_dim))
        if bias:
            self.linear_bias = nn.Parameter(torch.empty(linear_out_features))
        else:
            self.register_parameter('linear_bias', None)

        # Initialize parameters to match reference nn.MultiheadAttention and nn.Linear defaults
        # MultiheadAttention: in_proj_weight is xavier_uniform, in_proj_bias zeros,
        # out_proj.weight and bias from NonDynamicallyQuantizableLinear (xavier_uniform and zeros)
        # Linear: weight xavier_uniform, bias zeros
        nn.init.xavier_uniform_(self.attention_in_proj_weight)
        nn.init.zeros_(self.attention_in_proj_bias)
        nn.init.xavier_uniform_(self.attention_out_proj_weight)
        nn.init.zeros_(self.attention_out_proj_bias)
        nn.init.xavier_uniform_(self.linear_weight)
        if bias:
            nn.init.zeros_(self.linear_bias)

        # Register state_dict keys to match reference exactly
        # Reference keys: attention.in_proj_weight, attention.in_proj_bias,
        # attention.out_proj.weight, attention.out_proj.bias, linear.weight, linear.bias
        self._parameter_name_mapping = {
            'attention.in_proj_weight': 'attention_in_proj_weight',
            'attention.in_proj_bias': 'attention_in_proj_bias',
            'attention.out_proj.weight': 'attention_out_proj_weight',
            'attention.out_proj.bias': 'attention_out_proj_bias',
            'linear.weight': 'linear_weight',
            'linear.bias': 'linear_bias',
        }

    def _get_mapped_state_dict(self):
        # Override state_dict to use reference key names
        state = super().state_dict()
        mapped = {}
        for ref_key, param_key in self._parameter_name_mapping.items():
            if param_key in state:
                mapped[ref_key] = state[param_key]
            else:
                # For linear_bias when bias=False, reference would not have the key? Actually reference Model always has bias=True in current init? 
                # The reference constructor accepts bias, but default is True. If bias=False, reference Linear would have no bias parameter.
                # To be safe, only include if exists.
                pass
        return mapped

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Adjust for prefix (state_dict keys may include module prefixes if nested; but at top level prefix='')
        # We need to map reference keys to our parameter names.
        mapped_state = {}
        for ref_key, param_key in self._parameter_name_mapping.items():
            full_ref_key = prefix + ref_key
            if full_ref_key in state_dict:
                mapped_state[param_key] = state_dict[full_ref_key]
        # Call super with mapped_state and same prefix (empty after mapping)
        super()._load_from_state_dict(mapped_state, '', local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, query, key, value):
        # MultiheadAttention forward using custom op
        attn_result = cuda_extension.custom_multihead_attention(
            query, key, value,
            self.attention_in_proj_weight,
            self.attention_in_proj_bias,
            self.attention_out_proj_weight,
            self.attention_out_proj_bias,
            self.embed_dim,
            self.num_heads,
            True,  # need_weights=True (reference default)
            True   # average_attn_weights=True (reference default)
        )
        attn_output, attn_weights = attn_result  # tuple unpacking

        # Concatenate attn_output with itself along last dim
        x = cuda_extension.custom_cat(attn_output, attn_output, -1)

        # GLU
        x = cuda_extension.custom_glu(x, -1)

        # Linear
        if self.linear_bias is not None:
            x = cuda_extension.custom_linear(x, self.linear_weight, self.linear_bias)
        else:
            x = cuda_extension.custom_linear(x, self.linear_weight, None)

        return x

    def debug_forward(self, query, key, value):
        # Replicate forward but yield checkpoint values in order
        attn_result = cuda_extension.custom_multihead_attention(
            query, key, value,
            self.attention_in_proj_weight,
            self.attention_in_proj_bias,
            self.attention_out_proj_weight,
            self.attention_out_proj_bias,
            self.embed_dim,
            self.num_heads,
            True,
            True
        )
        attn_output, attn_weights = attn_result

        # Yield step_001 value as tuple (attn_output, attn_weights)
        yield ('step_001', (attn_output, attn_weights))

        x = cuda_extension.custom_cat(attn_output, attn_output, -1)
        yield ('step_002', x)

        x = cuda_extension.custom_glu(x, -1)
        yield ('step_003', x)

        if self.linear_bias is not None:
            x = cuda_extension.custom_linear(x, self.linear_weight, self.linear_bias)
        else:
            x = cuda_extension.custom_linear(x, self.linear_weight, None)
        yield ('step_004', x)

        yield ('step_005', x)
