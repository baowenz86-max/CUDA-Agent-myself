#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void softshrink_launcher(float* output, const float* input, long long n, float lambd, cudaStream_t stream);

torch::Tensor softshrink_forward(torch::Tensor x, double lambd) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must have at least one element");
    TORCH_CHECK(lambd >= 0, "lambd must be non-negative");

    torch::Tensor x_cuda;
    if (x.is_cuda()) {
        x_cuda = x.contiguous();
    } else {
        x_cuda = x.to(torch::kCUDA);
    }

    auto output = torch::empty_like(x_cuda);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softshrink_launcher(
        output.data_ptr<float>(),
        x_cuda.data_ptr<float>(),
        static_cast<long long>(x_cuda.numel()),
        static_cast<float>(lambd),
        stream
    );

    return output;
}

void register_softshrink(pybind11::module& m) {
    m.def("softshrink_forward", &softshrink_forward,
          "Softshrink forward (CUDA)",
          py::arg("x"),
          py::arg("lambd") = 0.5);
}

REGISTER_BINDING(softshrink, register_softshrink);
