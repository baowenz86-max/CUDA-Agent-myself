#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input, float* output, long long total_elements, cudaStream_t stream);
extern "C" void instance_norm3d_forward_launcher(const float* input, float* output,
                                                 int N, int C, long long spatial_size,
                                                 long long channel_stride, float eps,
                                                 cudaStream_t stream);

static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );
    check_cuda_error();

    return output;
}

torch::Tensor instance_norm3d_forward(torch::Tensor x, int64_t num_features = 32, double eps = 1e-5) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    long long spatial_size = x.size(2) * x.size(3) * x.size(4);
    long long channel_stride = spatial_size;

    TORCH_CHECK(C == num_features, "num_features must match input channel dimension");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    instance_norm3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, spatial_size, channel_stride,
        static_cast<float>(eps),
        stream
    );
    check_cuda_error();

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
    m.def("instance_norm3d_forward", &instance_norm3d_forward, "InstanceNorm3d forward",
          py::arg("x"),
          py::arg("num_features") = 32,
          py::arg("eps") = 1e-5);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
