import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim, max_threshold, any_threshold):
        super().__init__()
        self.max_threshold = max_threshold
        self.any_threshold = any_threshold

    def forward(self, x):
        # Apply torch.max to x with max_threshold
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        max_mask = max_result > self.max_threshold

        # Apply torch.any to x with any_threshold
        any_mask = torch.any(x > self.any_threshold, dim=1, keepdim=True)

        # Combine the two masks
        combined_mask = torch.logical_and(max_mask, any_mask)

        # Apply the combined mask to x
        result = torch.where(combined_mask, x, torch.zeros_like(x))
        return result


input_dim = 512
max_threshold = 0.5
any_threshold = 0.2


def get_inputs():
    return [torch.randn(64, input_dim, 128, 128)]


def get_init_inputs():
    return [input_dim, max_threshold, any_threshold]