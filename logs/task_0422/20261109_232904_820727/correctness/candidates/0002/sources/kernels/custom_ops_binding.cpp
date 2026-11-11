// Binding for custom max_dim and mse_loss operators
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim_launcher(const float* x,
                                        float* values,
                                        int64_t* indices,
                                        int64_t dim,
                                        bool keepdim,
                                        int64_t B,
                                        int64_t C,
                                        int64_t H,
                                        int64_t W,
                                        cudaStream_t stream);

extern "C" void custom_mse_loss_launcher(const float* input,
                                          const float* target,
                                          float* output,
                                          int64_t B,
                                          int64_t H,
                                          int64_t W,
                                          int64_t num_elements,
                                          cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> custom_max_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, C, H, W)");
    TORCH_CHECK(dim == 1, "only dim=1 supported");
    TORCH_CHECK(keepdim == true, "only keepdim=true supported");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensors with shape [B, 1, H, W]
    auto values = torch::empty({B, 1, H, W}, x.options());
    auto indices = torch::empty({B, 1, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_max_dim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dim,
        keepdim,
        B,
        C,
        H,
        W,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_dim kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

torch::Tensor custom_mse_loss(torch::Tensor input, torch::Tensor target) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (B, 1, H, W)");
    TORCH_CHECK(target.dim() == 4, "target must be 4D (B, 1, 1, 1)");
    TORCH_CHECK(input.size(0) == target.size(0), "batch size mismatch");
    TORCH_CHECK(input.size(1) == 1, "input channel must be 1");
    TORCH_CHECK(target.size(1) == 1 && target.size(2) == 1 && target.size(3) == 1, "target must have shape [B,1,1,1]");

    int64_t B = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t num_elements = B * H * W;

    // Allocate scalar output tensor
    auto output = torch::empty({}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (launcher will initialize output to 0 via cudaMemsetAsync)
    custom_mse_loss_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        H,
        W,
        num_elements,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_mse_loss kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim", &custom_max_dim, "Custom max over dim=1 with keepdim=true",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("custom_mse_loss", &custom_mse_loss, "Custom MSE loss",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
