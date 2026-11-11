#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" cudaError_t mean_dim_1_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           int stride_n, int stride_c, int stride_h, int stride_w,
                                           cudaStream_t stream);

torch::Tensor mean_dim_1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Input shape: [N, C, H, W]
    TORCH_CHECK(x.dim() == 4, "Input must have 4 dimensions");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output shape: [N, H, W]
    auto output = torch::empty({N, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides in elements
    int64_t stride_n = x.stride(0);
    int64_t stride_c = x.stride(1);
    int64_t stride_h = x.stride(2);
    int64_t stride_w = x.stride(3);

    cudaError_t err = mean_dim_1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(stride_n), static_cast<int>(stride_c), static_cast<int>(stride_h), static_cast<int>(stride_w),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "mean_dim_1 launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_mean_dim_1(pybind11::module& m) {
    m.def("mean_dim_1", &mean_dim_1_forward,
          "Compute mean along dimension 1 (channels) of a 4D tensor",
          py::arg("x"));
}

REGISTER_BINDING(mean_dim_1, register_mean_dim_1);
