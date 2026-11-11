import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.to('cuda')
        return cuda_extension.embedding_gt_forward(self.embedding.weight, x, self.embedding.padding_idx, 0.5)

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.to('cuda')
        y = cuda_extension.embedding_gt_forward(self.embedding.weight, x, self.embedding.padding_idx, 0.5)
        yield ('step_001', y)


def get_inputs():
    return [torch.randint(0, 1000, (256, 2048), dtype=torch.int64)]


def get_init_inputs():
    return [1000, 300, 0]
