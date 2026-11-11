#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void matmul_launcher(const float* x, const float* y, float* out,
                                int M, int N, int K, cudaStream_t stream);
extern "C" void softmax_dim0_launcher(const float* input, float* output,
                                      int rows, int cols, cudaStream_t stream);

// Helper to check CUDA errors after kernel launches
static void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error: ", cudaGetErrorString(err));
}

// matmul binding: x [M,K], y [K,N] -> out [M,N]
torch::Tensor matmul(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "matmul: x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "matmul: y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "matmul: x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "matmul: y must be float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "matmul: inputs must be 2D");
    TORCH_CHECK(x.size(1) == y.size(0), "matmul: inner dimensions must match");

    // Ensure contiguous (row-major) for cuBLAS
    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto y_contig = y.is_contiguous() ? y : y.contiguous();

    int M = x_contig.size(0);
    int K = x_contig.size(1);
    int N = y_contig.size(1);

    auto out = torch::empty({M, N}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    matmul_launcher(
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K,
        stream
    );

    check_cuda_error("matmul");
    return out;
}

// softmax_dim0 binding: input shape [rows, cols]
torch::Tensor softmax_dim0(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "softmax_dim0: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax_dim0: x must be float32");
    TORCH_CHECK(x.dim() == 2, "softmax_dim0: input must be 2D");

    // Ensure contiguous (kernel assumes row-major contiguous)
    auto x_contig = x.is_contiguous() ? x : x.contiguous();

    int rows = x_contig.size(0);
    int cols = x_contig.size(1);

    auto out = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim0_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        rows, cols,
        stream
    );

    check_cuda_error("softmax_dim0");
    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("matmul", &matmul, "Matrix multiplication using cuBLAS",
          py::arg("x"), py::arg("y"));
    m.def("softmax_dim0", &softmax_dim0, "Softmax along dimension 0",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
