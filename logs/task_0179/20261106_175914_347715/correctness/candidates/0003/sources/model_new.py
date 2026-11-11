import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, padding):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        # 参考模型无参数，保持 state_dict 为空

    def forward(self, x):
        # 将输入移到 CUDA（参考执行设备为 cuda）
        x = cuda_extension.move_to_cuda(x)
        # stft + abs
        x = cuda_extension.stft_abs(x, self.n_fft, self.hop_length, self.win_length)
        # reflection pad 2d
        padding_tensor = torch.tensor(self.padding, dtype=torch.int64, device='cpu')
        x = cuda_extension.reflection_pad_2d(x, padding_tensor)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.move_to_cuda(x)
        x = cuda_extension.stft_abs(x, self.n_fft, self.hop_length, self.win_length)
        yield 'step_001', x
        padding_tensor = torch.tensor(self.padding, dtype=torch.int64, device='cpu')
        x = cuda_extension.reflection_pad_2d(x, padding_tensor)
        yield 'step_002', x
        yield 'step_003', x
