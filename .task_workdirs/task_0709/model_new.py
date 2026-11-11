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
        # x shape: [batch_size, ...]
        batch_size = x.shape[0]
        chunk_size = batch_size // self.num_images
        xs = (x[0:chunk_size], x[chunk_size:2*chunk_size])
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        # concatenate histograms manually without torch.stack/cat
        flat_hist = torch.empty(self.num_images * self.num_bins, device=x.device, dtype=x.dtype)
        for i, hist in enumerate(ys):
            flat_hist[i * self.num_bins:(i + 1) * self.num_bins] = hist
        y = cuda_extension.cat_histc(flat_hist, self.num_images, self.num_bins)
        return y

    def debug_forward(self, x):
        batch_size = x.shape[0]
        chunk_size = batch_size // self.num_images
        xs = (x[0:chunk_size], x[chunk_size:2*chunk_size])
        yield 'step_001', xs
        ys = [cuda_extension.compute_histc(chunk, self.num_bins, self.bins_range[0], self.bins_range[1]) for chunk in xs]
        yield 'step_002', ys
        flat_hist = torch.empty(self.num_images * self.num_bins, device=x.device, dtype=x.dtype)
        for i, hist in enumerate(ys):
            flat_hist[i * self.num_bins:(i + 1) * self.num_bins] = hist
        y = cuda_extension.cat_histc(flat_hist, self.num_images, self.num_bins)
        yield 'step_003', y
        yield 'step_004', y
