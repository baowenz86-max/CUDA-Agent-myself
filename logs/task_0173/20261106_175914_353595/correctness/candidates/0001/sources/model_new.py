import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.einsum_input = "bhwc->bwh"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.einsum_bhwc_to_bwh(x)
        x = cuda_extension.abs_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.einsum_bhwc_to_bwh(x)
        yield ("step_001", x)
        x = cuda_extension.abs_forward(x)
        yield ("step_002", x)
        yield ("step_003", x)
