import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, stride, padding):
        super().__init__()
        # Store as plain attributes; no state_dict entries.
        self.kernel_size1 = tuple(kernel_size1)
        self.kernel_size2 = tuple(kernel_size2)
        self.stride = int(stride)
        self.padding = tuple(padding)

    def forward(self, x):
        # Use custom op that computes both avg pools sequentially according to reference.
        x = cuda_extension.avg_pool3d_double_forward(
            x,
            self.kernel_size1[0], self.kernel_size1[1], self.kernel_size1[2],
            self.kernel_size2[0], self.kernel_size2[1], self.kernel_size2[2],
            self.stride,
            self.padding[0], self.padding[1], self.padding[2]
        )
        return x

    def debug_forward(self, x):
        # For debugging, we need to produce checkpoints step_001 and step_002 separately.
        # We can call two custom ops: one for each pooling stage.
        # To maintain consistency with forward, we could also call the double op and then
        # reconstruct intermediate? But the contract requires debug_forward to yield the
        # intermediate value exactly. So we need a separate custom op for single avg_pool3d.
        # However, the operator_contract must cover all calls. We will define two ops:
        # avg_pool3d_forward (single) and avg_pool3d_double_forward (double).
        # For forward, we use double for potential fusion; for debug, we use single twice.
        # This is allowed as long as both are in the contract.
        step1 = cuda_extension.avg_pool3d_forward(
            x,
            self.kernel_size1[0], self.kernel_size1[1], self.kernel_size1[2],
            self.stride,
            self.padding[0], self.padding[1], self.padding[2]
        )
        yield ('step_001', step1)
        step2 = cuda_extension.avg_pool3d_forward(
            step1,
            self.kernel_size2[0], self.kernel_size2[1], self.kernel_size2[2],
            self.stride,
            self.padding[0], self.padding[1], self.padding[2]
        )
        yield ('step_002', step2)
        yield ('step_003', step2)
