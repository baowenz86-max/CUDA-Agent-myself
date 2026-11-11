#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(const float* input,
                                        float* output,
                                        int outer,
                                        int inner,
                                        int innermost,
                                        long long stride0,
                                        long long stride1,
                                        long long stride2,
                                        cudaStream_t stream);

extern "C" void custom_sub_scalar_launcher(const float* input,
                                           float* output,
                                           long long total_elements,
                                           float scalar,
                                           cudaStream_t stream);

// Wrapper for custom_softmax
torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");
    // The launcher currently supports dim=1 only; contract uses dim=1.
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for custom_softmax");

    auto output = torch::empty_like(x);

    int outer = x.size(0);
    int inner = x.size(1);
    int innermost = x.size(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer,
        inner,
        innermost,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        stream
    );

    return output;
}

// Wrapper for custom_sub_scalar
torch::Tensor custom_sub_scalar_forward(torch::Tensor x, double other) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");

    auto output = torch::empty_like(x);

    long long total_elements = x.numel();
    float scalar = static_cast<float>(other);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sub_scalar_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        scalar,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax_forward,
          "Softmax along dimension 1 of a 3D tensor",
          py::arg("x"),
          py::arg("dim"));
    m.def("custom_sub_scalar", &custom_sub_scalar_forward,
          "Elementwise subtraction of a scalar",
          py::arg("x"),
          py::arg("other"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
