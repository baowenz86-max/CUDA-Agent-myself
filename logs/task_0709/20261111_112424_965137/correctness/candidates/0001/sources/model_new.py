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
        xs = torch.chunk(x, chunks=self.num_images, dim=0)
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        y = cuda_extension.cat_histc(torch.stack(ys), self.num_images, self.num_bins)
        return y

    def debug_forward(self, x):
        xs = torch.chunk(x, chunks=self.num_images, dim=0)
        yield 'step_001', xs
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        yield 'step_002', ys
        y = cuda_extension.cat_histc(torch.stack(ys), self.num_images, self.num_bins)
        yield 'step_003', y
        yield 'step_004', y
