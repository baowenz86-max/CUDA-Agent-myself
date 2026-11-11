#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_dim_launcher(const float* input,
                                          float* output,
                                          int rows, int cols,
                                          cudaStream_t stream);

extern "C" void cat_dim_launcher(const float* input_a,
                                  const float* input_b,
                                  float* output,
                                  int rows_a, int rows_b, int cols,
                                  cudaStream_t stream);

// Wrapper for log_softmax along dim=0
torch::Tensor log_softmax_dim(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for log_softmax");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_dim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_dim kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for concatenation along dim=0
torch::Tensor cat_dim(torch::Tensor input_a, torch::Tensor input_b, int64_t dim) {
    TORCH_CHECK(input_a.is_cuda() && input_b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input_a.dtype() == torch::kFloat32 && input_b.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(input_a.is_contiguous() && input_b.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(input_a.dim() == 2 && input_b.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for cat");
    TORCH_CHECK(input_a.size(1) == input_b.size(1), "Inputs must have same number of columns");

    int rows_a = input_a.size(0);
    int rows_b = input_b.size(0);
    int cols = input_a.size(1);

    auto output = torch::empty({rows_a + rows_b, cols}, input_a.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cat_dim_launcher(
        input_a.data_ptr<float>(),
        input_b.data_ptr<float>(),
        output.data_ptr<float>(),
        rows_a, rows_b, cols,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cat_dim kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_dim", &log_softmax_dim, "Log softmax along dim=0",
          pybind11::arg("input"), pybind11::arg("dim"));
    m.def("cat_dim", &cat_dim, "Concatenate along dim=0",
          pybind11::arg("input_a"), pybind11::arg("input_b"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
