#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void mse_sum_launcher(float* out,
                                 const float* x,
                                 const float* y,
                                 long n,
                                 cudaStream_t stream);

torch::Tensor mse_sum_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();

    long n = x_contig.numel();
    TORCH_CHECK(n > 0, "input tensors must not be empty");

    auto out = torch::empty({}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mse_sum_launcher(
        out.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        n,
        stream
    );

    return out;
}

void register_mse_sum_forward(pybind11::module& m) {
    m.def("mse_sum_forward", &mse_sum_forward,
          "Compute MSE sum (mean squared error) between two CUDA float32 tensors",
          py::arg("x"), py::arg("y"));
}

REGISTER_BINDING(mse_sum_forward, register_mse_sum_forward);
