import torch
import torch.nn as nn

try:
    import cuda_extension
except ImportError:
    cuda_extension = None


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        if cuda_extension is not None:
            embedded = cuda_extension.custom_embedding_forward(self.embedding.weight, x)
            target = cuda_extension.custom_randn_like(embedded)
            loss = cuda_extension.custom_smooth_l1_loss_forward(embedded, target)
            return loss
        else:
            # Fallback for environments without compiled extension (not used in verification)
            raise RuntimeError('cuda_extension is not available')

    def debug_forward(self, *inputs):
        x = inputs[0]
        embedded = cuda_extension.custom_embedding_forward(self.embedding.weight, x)
        target = cuda_extension.custom_randn_like(embedded)
        loss = cuda_extension.custom_smooth_l1_loss_forward(embedded, target)
        yield 'step_001', loss
