#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu
extern "C" void instance_norm_forward_launcher(
    const float* input,
    float* output,
    double* mean,
    double* var,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const int64_t* stride,
    int64_t total_elems,
    float eps,
    cudaStream_t stream);

// PyTorch wrapper for instance_norm_forward
torch::Tensor instance_norm_forward(torch::Tensor x, int64_t num_features, double eps) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    TORCH_CHECK(C == num_features, "Input channel dimension must match num_features");

    // Allocate output tensor with same shape and strides as input to support arbitrary strides
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Total number of elements
    int64_t total_elems = x.numel();
    if (total_elems == 0) {
        return output;
    }

    // Number of per-(N,C) channels
    int64_t num_channels_total = N * C;

    // Allocate temporary mean and var buffers (double) on device
    auto mean = torch::empty({num_channels_total}, x.options().dtype(torch::kFloat64));
    auto var = torch::empty({num_channels_total}, x.options().dtype(torch::kFloat64));

    // Prepare stride array on device (int64_t)
    auto stride_tensor = torch::empty({5}, x.options().dtype(torch::kInt64));
    int64_t stride_host[5];
    stride_host[0] = x.stride(0);
    stride_host[1] = x.stride(1);
    stride_host[2] = x.stride(2);
    stride_host[3] = x.stride(3);
    stride_host[4] = x.stride(4);
    cudaMemcpyAsync(stride_tensor.data_ptr<int64_t>(), stride_host, 5 * sizeof(int64_t), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream().stream());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    instance_norm_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<double>(),
        var.data_ptr<double>(),
        N,
        C,
        D,
        H,
        W,
        stride_tensor.data_ptr<int64_t>(),
        total_elems,
        static_cast<float>(eps),
        stream
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in instance_norm_forward_launcher: ", cudaGetErrorString(err));

    return output;
}

// Register binding
void register_instance_norm(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "Instance Normalization forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("num_features"),
          pybind11::arg("eps"));
}

REGISTER_BINDING(instance_norm, register_instance_norm);
