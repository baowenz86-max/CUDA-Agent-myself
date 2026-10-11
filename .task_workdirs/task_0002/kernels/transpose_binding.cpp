#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from transpose_kernel.cu
extern "C" void transpose_1_2_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

// PyTorch wrapper for transpose_1_2
torch::Tensor transpose_1_2_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty({N, H, C, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    transpose_1_2_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

// Registration function
void register_transpose(pybind11::module& m) {
    m.def("transpose_1_2_forward", &transpose_1_2_forward, "Transpose dims 1 and 2",
          py::arg("input"));
}

REGISTER_BINDING(transpose, register_transpose);
