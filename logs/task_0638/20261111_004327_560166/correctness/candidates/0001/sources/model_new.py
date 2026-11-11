import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_embeddings_1, num_embeddings_2, embedding_dim):
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings_1, embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings_2, embedding_dim)

    def forward(self, x1, x2):
        x1 = x1.cuda()
        x2 = x2.cuda()
        emb1 = cuda_extension.embedding1_forward(x1, self.embedding1.weight)
        emb2 = cuda_extension.embedding2_forward(x2, self.embedding2.weight)
        return cuda_extension.add_tensors(emb1, emb2)

    def debug_forward(self, x1, x2):
        x1 = x1.cuda()
        x2 = x2.cuda()
        emb1 = cuda_extension.embedding1_forward(x1, self.embedding1.weight)
        emb2 = cuda_extension.embedding2_forward(x2, self.embedding2.weight)
        result = cuda_extension.add_tensors(emb1, emb2)
        yield "step_001", result
