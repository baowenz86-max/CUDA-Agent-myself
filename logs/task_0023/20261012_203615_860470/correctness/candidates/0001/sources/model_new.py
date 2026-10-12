import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    """Optimized model using custom CUDA kernels.

    Custom ops contract:
        cuda_extension.median_dim0_forward(x: Tensor) -> Tensor
            x: float32 CUDA tensor of shape [N, C, H, W]
            returns median along dim=0 (lower median for even N), shape [C, H, W]
        cuda_extension.reflection_pad2d_forward(x: Tensor, pad_size: int) -> Tensor
            x: float32 CUDA tensor of shape [C, H, W]
            returns reflection padded tensor with pad_size on all sides,
            output shape [C, H+2*pad_size, W+2*pad_size]
    """

    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = int(pad_size)

    def forward(self, x):
        # Step 1: median along batch dimension
        x = cuda_extension.median_dim0_forward(x)

        # Step 2: meshgrid on CPU as in reference (not performance-critical)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, x.size(0)),
            torch.linspace(0, 1, x.size(1)),
            indexing="ij",
        )

        # Step 3: reflection padding
        x = cuda_extension.reflection_pad2d_forward(x, self.pad_size)
        return x

    def debug_forward(self, *inputs):
        """Yield checkpoints matching the reference trace, including final return."""
        x = inputs[0]
        x = cuda_extension.median_dim0_forward(x)
        yield "step_001", x

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, x.size(0)),
            torch.linspace(0, 1, x.size(1)),
            indexing="ij",
        )
        yield "step_002", (grid_x, grid_y)

        x = cuda_extension.reflection_pad2d_forward(x, self.pad_size)
        yield "step_003", x

        yield "step_004", x
