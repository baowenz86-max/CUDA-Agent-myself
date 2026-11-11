import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        # Match original state_dict key "embedding.weight" by using a submodule Embedding
        # Only weights are needed; padding_idx is ignored except for state_dict compatibility.
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        # Move indices to CUDA if necessary; custom op expects CUDA int64 tensor.
        if not x.is_cuda:
            x = x.cuda()
        indices = x.to(torch.int64)
        weight = self.embedding.weight
        if not weight.is_cuda:
            weight = weight.cuda()

        # Step 1: embedding lookup
        embedded = cuda_extension.embedding_forward(indices, weight)

        # Step 2: mean over dim=1
        result = cuda_extension.mean_dim1_forward(embedded)

        return result

    def debug_forward(self, *inputs):
        # inputs[0] is x
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        indices = x.to(torch.int64)
        weight = self.embedding.weight
        if not weight.is_cuda:
            weight = weight.cuda()

        embedded = cuda_extension.embedding_forward(indices, weight)
        yield ("step_001", embedded)

        mean_result = cuda_extension.mean_dim1_forward(embedded)
        yield ("step_002", mean_result)

        yield ("step_003", mean_result)
