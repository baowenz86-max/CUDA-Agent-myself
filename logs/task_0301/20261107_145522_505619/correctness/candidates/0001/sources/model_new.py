import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return cuda_extension.batched_matmul(x, y, False, False)

    def debug_forward(self, *inputs):
        for checkpoint_id, value in self._debug_checkpoints(inputs):
            yield checkpoint_id, value

    def _debug_checkpoints(self, inputs):
        x, y = inputs
        result = cuda_extension.batched_matmul(x, y, False, False)
        yield 'step_001', result
