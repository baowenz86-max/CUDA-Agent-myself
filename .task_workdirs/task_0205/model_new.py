import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, mask_value):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mask_value = mask_value

    def forward(self, x):
        # Step 1: adaptive average pooling
        x = cuda_extension.aap1d_forward(x, self.output_size)
        # Step 2: create scalar boolean mask on same device
        mask = torch.tensor(self.mask_value, dtype=torch.bool, device=x.device)
        # Step 3: masked_select
        return cuda_extension.masked_select_forward(x, mask)

    def debug_forward(self, x):
        # Debug: yield each checkpoint in order
        x = cuda_extension.aap1d_forward(x, self.output_size)
        yield 'step_001', x
        mask = torch.tensor(self.mask_value, dtype=torch.bool, device=x.device)
        yield 'step_002', mask
        result = cuda_extension.masked_select_forward(x, mask)
        yield 'step_003', result