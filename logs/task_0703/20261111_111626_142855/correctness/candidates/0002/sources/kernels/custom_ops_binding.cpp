#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim_023_launcher(const float* input, float* output,
                                      int N, int C, int H, int W,
                                      cudaStream_t stream);

extern "C" void histc_custom_launcher(const float* input, float* output,
                                      int C, int bins, double min_val, double max_val,
                                      cudaStream_t stream);

// Wrapper for mean over dims (0,2,3)
torch::Tensor mean_dim_023(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mean_dim_023: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim_023: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_dim_023: input must be a 4D tensor");
    TORCH_CHECK(x.is_contiguous(), "mean_dim_023: input must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({C}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim_023_launcher(x.data_ptr<float>(),
                         output.data_ptr<float>(),
                         N, C, H, W,
                         stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim_023 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for histogram
torch::Tensor histc_custom(torch::Tensor x, int64_t bins, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "histc_custom: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "histc_custom: input must be float32");
    TORCH_CHECK(x.dim() == 1, "histc_custom: input must be a 1D tensor");
    TORCH_CHECK(x.is_contiguous(), "histc_custom: input must be contiguous");
    TORCH_CHECK(bins > 0, "histc_custom: bins must be positive");

    int C = x.size(0);
    int64_t int_bins = bins;

    auto output = torch::empty({int_bins}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_custom_launcher(x.data_ptr<float>(),
                         output.data_ptr<float>(),
                         C, int_bins, min_val, max_val,
                         stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "histc_custom kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("mean_dim_023", &mean_dim_023, "mean over dims (0,2,3)",
          py::arg("x"));
    m.def("histc_custom", &histc_custom, "histogram custom",
          py::arg("x"), py::arg("bins"), py::arg("min_val"), py::arg("max_val"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);