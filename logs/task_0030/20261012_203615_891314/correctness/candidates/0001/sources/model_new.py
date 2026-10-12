import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, magnitudes, angles, query, key, value):
        complex_numbers = cuda_extension.polar_forward(magnitudes, angles)
        one_hot_encodings = cuda_extension.one_hot_forward(self.num_classes)
        attention_output = cuda_extension.sdpa_forward(query, key, value)
        return complex_numbers, one_hot_encodings, attention_output

    def debug_forward(self, magnitudes, angles, query, key, value):
        complex_numbers = cuda_extension.polar_forward(magnitudes, angles)
        yield 'step_001', complex_numbers
        one_hot_encodings = cuda_extension.one_hot_forward(self.num_classes)
        yield 'step_002', one_hot_encodings
        attention_output = cuda_extension.sdpa_forward(query, key, value)
        yield 'step_003', attention_output
        yield 'step_004', (complex_numbers, one_hot_encodings, attention_output)
