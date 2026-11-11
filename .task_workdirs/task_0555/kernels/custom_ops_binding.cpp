// PyTorch binding for custom_max_dim1 and custom_normalize
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim1_launcher(
    float* values,
    int64_t* indices,
    const float* input,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream
);

extern "C" void custom_normalize_launcher(
    float* output,
    const float* input,
    int N,
    int C,
    int H,
    int W,
    int dim,
    cudaStream_t stream
);

// Wrapper for custom_max_dim1: torch.max(x, dim=1, keepdim=True)
std::tuple<torch::Tensor, torch::Tensor> custom_max_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_max_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_dim1: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_max_dim1: input must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "custom_max_dim1: invalid shape");
    TORCH_CHECK(C <= 32, "custom_max_dim1: channel dimension should be 32"); // optional, but matches contract

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_dim1_launcher(
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

// Wrapper for custom_normalize: F.normalize(x, p=2.0, dim=dim)
torch::Tensor custom_normalize(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_normalize: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_normalize: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_normalize: input must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "custom_normalize: invalid shape");
    TORCH_CHECK(dim >= 0 && dim < 4, "custom_normalize: dim out of range");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_normalize_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(dim),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_normalize kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1, "Compute torch.max(x, dim=1, keepdim=True) returning tuple(values, indices)", py::arg("x"));
    m.def("custom_normalize", &custom_normalize, "L2 normalize along given dimension", py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
