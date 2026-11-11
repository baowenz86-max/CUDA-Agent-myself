import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bij,bjk->bik", x, x)
        x = torch.bmm(x, x)
        return x


def get_inputs():
    batch_size, in_features = 4, 1280
    x = torch.randn(batch_size, in_features, in_features)
    return [x]


def get_init_inputs():
    return []