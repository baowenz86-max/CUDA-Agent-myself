import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, pad_size, pad_mode):
        super().__init__()
        # Store pad parameters for use in custom ops; no torch modules or parameters
        self.pad_size = tuple(pad_size)
        self.pad_mode = pad_mode
        self.channels = channels

    def forward(self, x):
        # Move input to CUDA and ensure float32 contiguous if necessary
        # Note: .cuda() and .contiguous() are allowed tensor methods, not torch operators
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        # ReflectionPad2d with pad_size=(1,2,1,2)
        x = cuda_extension.reflection_pad2d_forward(
            x, self.pad_size[0], self.pad_size[1], self.pad_size[2], self.pad_size[3]
        )

        # Argsort along dim=1
        x = cuda_extension.argsort_dim1_forward(x)

        return x

    def debug_forward(self, *inputs):
        # Unpack single input tensor as in forward
        x = inputs[0]

        # Replicate the exact computation and yield checkpoints
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        x = cuda_extension.reflection_pad2d_forward(
            x, self.pad_size[0], self.pad_size[1], self.pad_size[2], self.pad_size[3]
        )
        yield ('step_001', x)

        x = cuda_extension.argsort_dim1_forward(x)
        yield ('step_002', x)

        yield ('step_003', x)
