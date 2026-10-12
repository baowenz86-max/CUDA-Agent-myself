#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

__global__ void triu_kernel(float* output, const float* input, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;

    // Compute position (b,c,h,w) from linear index
    int w = idx % width;
    int h = (idx / width) % height;
    // For triu: keep if w >= h, else set to 0
    if (w >= h) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

torch::Tensor triu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto output = torch::empty_like(input);
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int total = batch * channels * height * width;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_kernel<<<blocks, threads, 0, stream>>>(output.data_ptr<float>(), input.data_ptr<float>(),
                                                batch, channels, height, width);
    return output;
}

void register_triu_forward(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Upper triangular operation");
}

REGISTER_BINDING(triu_forward, register_triu_forward);
