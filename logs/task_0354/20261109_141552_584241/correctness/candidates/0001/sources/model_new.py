import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size_h, pad_size_w):
        super().__init__()
        self.pad_size_h = int(pad_size_h)
        self.pad_size_w = int(pad_size_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.replication_pad2d_forward(
            x, self.pad_size_h, self.pad_size_h, self.pad_size_w, self.pad_size_w
        )

    def debug_forward(self, x: torch.Tensor):
        y = cuda_extension.replication_pad2d_forward(
            x, self.pad_size_h, self.pad_size_h, self.pad_size_w, self.pad_size_w
        )
        yield "step_001", y
