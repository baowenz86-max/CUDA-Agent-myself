import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, pool_size):
        super().__init__()
        # Preserve state_dict: AdaptiveAvgPool1d has no learnable parameters, so using a plain attribute is safe.
        # However, if future versions require module structure, we replicate with a placeholder.
        self.stack = nn.AdaptiveAvgPool1d(output_size)
        self.pool_size = pool_size

    def forward(self, x):
        # Move input to CUDA if not already (match execution device)
        if not x.is_cuda:
            x = x.cuda()
        
        # Custom adaptive avg pool1d
        x = cuda_extension.adaptive_avg_pool1d(x, self.stack.output_size)
        
        # Custom replication along dim=1 (stack)
        x = cuda_extension.repeat_interleave_like(x, self.pool_size, 1)
        
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        
        # Step 1: AdaptiveAvgPool1d
        x = cuda_extension.adaptive_avg_pool1d(x, self.stack.output_size)
        yield ('step_001', x)
        
        # Step 2: Stack copies
        x = cuda_extension.repeat_interleave_like(x, self.pool_size, 1)
        yield ('step_002', x)
        
        # Step 3: return
        yield ('step_003', x)
