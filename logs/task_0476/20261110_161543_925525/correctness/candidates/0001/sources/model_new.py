import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size, roll_shifts):
        super().__init__()
        self.pad_size = tuple(pad_size)
        self.roll_shifts = tuple(roll_shifts)

    def forward(self, x):
        # Ensure input is on CUDA and float32 as required by custom ops
        x = x.cuda() if not x.is_cuda else x
        x = x.float() if x.dtype != torch.float32 else x

        # Pad tensor (left, right, top, bottom)
        pad_tensor = torch.tensor(self.pad_size, dtype=torch.int64, device='cpu')
        x = cuda_extension.custom_reflection_pad2d(x, pad_tensor)

        # Roll tensor on dims (2,3) with shifts
        shifts_tensor = torch.tensor(self.roll_shifts, dtype=torch.int64, device='cpu')
        dims_tensor = torch.tensor((2, 3), dtype=torch.int64, device='cpu')
        x = cuda_extension.custom_roll(x, shifts_tensor, dims_tensor)

        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError("debug_forward expects exactly one input tensor")
        x = inputs[0]
        x = x.cuda() if not x.is_cuda else x
        x = x.float() if x.dtype != torch.float32 else x

        pad_tensor = torch.tensor(self.pad_size, dtype=torch.int64, device='cpu')
        x = cuda_extension.custom_reflection_pad2d(x, pad_tensor)
        yield 'step_001', x

        shifts_tensor = torch.tensor(self.roll_shifts, dtype=torch.int64, device='cpu')
        dims_tensor = torch.tensor((2, 3), dtype=torch.int64, device='cpu')
        x = cuda_extension.custom_roll(x, shifts_tensor, dims_tensor)
        yield 'step_002', x

        yield 'step_003', x
