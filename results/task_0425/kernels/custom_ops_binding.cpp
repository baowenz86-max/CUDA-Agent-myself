// Binding file for custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim_1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    cudaStream_t stream
);

extern "C" void triu_last_two_dims_launcher(
    const float* x,
    float* out,
    long long total_elements,
    int rows,
    int cols,
    int diagonal,
    cudaStream_t stream
);

// Wrapper for max along dimension 1 with keepdim
torch::Tensor max_dim_1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto sizes = x.sizes();
    long long N = sizes[0];
    long long C = sizes[1];
    long long H = 1, W = 1;
    for (int i = 2; i < sizes.size(); ++i) {
        H *= sizes[i];
    }
    if (sizes.size() >= 3) {
        H = sizes[2];
    }
    if (sizes.size() >= 4) {
        W = sizes[3];
    }

    // Create output shape: same as input but with dim 1 = 1
    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes[1] = 1;
    auto output = torch::empty(out_sizes, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_dim_1_keepdim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for triu on last two dimensions
torch::Tensor triu_last_two_dims(torch::Tensor x, int64_t diagonal = 0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto sizes = x.sizes();
    long long total_elements = x.numel();
    int rows = static_cast<int>(sizes[sizes.size() - 2]);
    int cols = static_cast<int>(sizes[sizes.size() - 1]);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_last_two_dims_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        rows,
        cols,
        static_cast<int>(diagonal),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim_1_keepdim", &max_dim_1_keepdim,
          "Compute max along dimension 1 with keepdim",
          pybind11::arg("x"));
    m.def("triu_last_two_dims", &triu_last_two_dims,
          "Apply upper triangular mask on last two dimensions",
          pybind11::arg("x"),
          pybind11::arg("diagonal") = 0);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
