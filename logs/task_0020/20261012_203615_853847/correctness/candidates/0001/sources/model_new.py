import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_embeddings, embedding_dim
    ):
        super().__init__()
        # 保持与原始 nn.ConvTranspose1d 和 nn.Embedding 相同的参数结构
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 缓存标量参数供自定义算子使用
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x, indices):
        # 步骤1: ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
        )
        # 步骤2: sign
        x = cuda_extension.sign_forward(x)
        # 步骤3: embedding
        embedded = cuda_extension.embedding_forward(
            indices,
            self.embedding.weight,
        )
        return x, embedded

    def debug_forward(self, x, indices):
        # 按 checkpoints 顺序 yield (id, value)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x
        x = cuda_extension.sign_forward(x)
        yield "step_002", x
        embedded = cuda_extension.embedding_forward(
            indices,
            self.embedding.weight,
        )
        yield "step_003", embedded
        result = (x, embedded)
        yield "step_004", result
        return  # debug_forward 是一个 generator，不需要显式返回 result
