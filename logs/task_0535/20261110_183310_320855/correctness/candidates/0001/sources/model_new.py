import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_values, _ = torch.return_types.max((x, torch.empty_like(x, dtype=torch.int64)))
        return max_values

    def debug_forward(self, *inputs):
        x = inputs[0]
        values, indices = torch.return_types.max((x, torch.empty_like(x, dtype=torch.int64)))
        yield 'step_001', (values, indices)
        yield 'step_002', values
