import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, batch_size, input_dim, mat1_dim, mat2_dim):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.mat1_dim = mat1_dim
        self.mat2_dim = mat2_dim

    def forward(self, input, mat1, mat2):
        baddbmm_result = cuda_extension.baddbmm_custom(input, mat1, mat2)
        equal_result = cuda_extension.equal_custom(baddbmm_result, input)
        return equal_result, baddbmm_result

    def debug_forward(self, *inputs):
        input, mat1, mat2 = inputs
        baddbmm_result = cuda_extension.baddbmm_custom(input, mat1, mat2)
        yield 'step_001', baddbmm_result
        equal_result = cuda_extension.equal_custom(baddbmm_result, input)
        yield 'step_002', equal_result
        return_tuple = (equal_result, baddbmm_result)
        yield 'step_003', return_tuple