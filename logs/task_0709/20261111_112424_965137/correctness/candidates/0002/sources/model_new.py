import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_images, num_bins, bins_range):
        super().__init__()
        self.num_images = num_images
        self.num_bins = num_bins
        self.bins_range = bins_range

    def forward(self, x):
        chunk_size = x.size(0) // self.num_images
        xs = tuple(x[i * chunk_size:(i + 1) * chunk_size] for i in range(self.num_images))
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        y = torch.cat(ys, dim=0)
        return y

    def debug_forward(self, x):
        chunk_size = x.size(0) // self.num_images
        xs = tuple(x[i * chunk_size:(i + 1) * chunk_size] for i in range(self.num_images))
        yield 'step_001', xs
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        yield 'step_002', ys
        y = torch.cat(ys, dim=0)
        yield 'step_003', y
        yield 'step_004', y
