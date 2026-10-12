#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

__global__ void cast_float_kernel(float* output, const unsigned char* input, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = static_cast<float>(input[idx]);
}

torch::Tensor cast_float_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto output = torch::empty_like(input, input.options().dtype(torch::kFloat32));
    int total = input.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cast_float_kernel<<<blocks, threads, 0, stream>>>(output.data_ptr<float>(),
                                                      input.data_ptr<unsigned char>(), total);
    return output;
}

void register_cast_float_forward(pybind11::module& m) {
    m.def("cast_float_forward", &cast_float_forward, "Cast uint8 to float32");
}

REGISTER_BINDING(cast_float_forward, register_cast_float_forward);
