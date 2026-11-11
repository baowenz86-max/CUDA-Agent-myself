import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_result = cuda_extension.max_dim1_kernel(x)
        x = max_result.values
        x = cuda_extension.expm1_kernel(x)
        return x

    def debug_forward(self, x):
        import cuda_extension
        import torch
        from torch.utils.hooks import RemovableHandle

        checkpoints = [
            ('step_001', lambda: (self._max_step(x),)),
            ('step_002', lambda: (self._expm1_step(self._max_step(x)[0]),)),
            ('step_003', lambda: (self._expm1_step(self._max_step(x)[0]),)),
        ]

        for checkpoint_id, fn in checkpoints:
            yield checkpoint_id, fn()[0]

    def _max_step(self, x):
        return cuda_extension.max_dim1_kernel(x)

    def _expm1_step(self, x):
        return cuda_extension.expm1_kernel(x)
