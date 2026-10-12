import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_embeddings, embedding_dim
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x, indices):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x = cuda_extension.sign_forward(x)
        embedded = cuda_extension.embedding_forward(indices, self.embedding.weight)
        return x, embedded

    def debug_forward(self, x, indices):
        # Step 1: conv_transpose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield "step_001", x

        # Step 2: torch.sign
        x = cuda_extension.sign_forward(x)
        yield "step_002", x

        # Step 3: embedding
        embedded = cuda_extension.embedding_forward(indices, self.embedding.weight)
        yield "step_003", embedded

        # Step 4: return tuple
        result = (x, embedded)
        yield "step_004", result
