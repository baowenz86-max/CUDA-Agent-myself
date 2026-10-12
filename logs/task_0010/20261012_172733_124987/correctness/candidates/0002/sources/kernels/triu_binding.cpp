#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher defined in triu_forward.cu
extern "C" void triu_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream
);

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

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    return output;
}

void register_triu_forward(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Upper triangular operation");
}

REGISTER_BINDING(triu_forward, register_triu_forward);
