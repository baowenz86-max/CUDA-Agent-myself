#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
);

extern "C" void custom_scatter_dim_launcher(
    const float* input,
    const int64_t* index,
    const float* src,
    float* output,
    int64_t dim,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
);

torch::Tensor custom_mean_dim(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty({B, 1, H, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_dim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        B, C, H, W,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_mean_dim kernel launch failed");
    return output;
}

torch::Tensor custom_scatter_dim(torch::Tensor input, int64_t dim, torch::Tensor index, torch::Tensor src) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(index.dtype() == torch::kInt64, "index must be int64");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(index.dim() == 4, "index must be 4D");
    TORCH_CHECK(src.dim() == 4, "src must be 4D");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto input_contig = input.contiguous();
    auto index_contig = index.contiguous();
    auto src_contig = src.contiguous();

    auto output = torch::empty({B, C, H, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_scatter_dim_launcher(
        input_contig.data_ptr<float>(),
        index_contig.data_ptr<int64_t>(),
        src_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        B, C, H, W,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_scatter_dim kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim", &custom_mean_dim, "Mean along dim=1 with keepdim=True",
          py::arg("input"), py::arg("dim"));
    m.def("custom_scatter_dim", &custom_scatter_dim, "Scatter along dim=1",
          py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
