#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void cat_forward_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int N, int C1, int C2, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h,
    long long x1_stride_n, long long x1_stride_c, long long x1_stride_h,
    long long x2_stride_n, long long x2_stride_c, long long x2_stride_h,
    cudaStream_t stream
);

extern "C" void mse_loss_forward_launcher(
    float* result,
    const float* input,
    const float* target,
    long long total_elements,
    cudaStream_t stream
);

// PyTorch wrapper for cat_forward
torch::Tensor cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4, "x1 must be 4D");
    TORCH_CHECK(x2.dim() == 4, "x2 must be 4D");
    TORCH_CHECK(x1.size(0) == x2.size(0), "batch dim mismatch");
    TORCH_CHECK(x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3), "spatial dims mismatch");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int N = x1.size(0);
    int C1 = x1.size(1);
    int C2 = x2.size(1);
    int H = x1.size(2);
    int W = x1.size(3);
    int C_out = C1 + C2;

    // Allocate output tensor on CUDA, float32, contiguous
    auto out = torch::empty({N, C_out, H, W}, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with pointers and strides
    cat_forward_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        N, C1, C2, H, W,
        out.stride(0), out.stride(1), out.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        x2.stride(0), x2.stride(1), x2.stride(2),
        stream
    );

    return out;
}

torch::Tensor mse_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have the same shape");

    long long total_elements = input.numel();
    if (total_elements == 0) {
        // Return nan scalar tensor to mimic PyTorch behavior for empty MSE
        return torch::full({}, NAN, input.options());
    }

    // Allocate scalar output tensor on CUDA
    auto result = torch::empty({}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mse_loss_forward_launcher(
        result.data_ptr<float>(),
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        total_elements,
        stream
    );

    return result;
}

void register_custom_ops(pybind11::module& m) {
    m.def("cat_forward", &cat_forward,
          "Concatenate two tensors along dim=1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("mse_loss_forward", &mse_loss_forward,
          "Mean squared error loss",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
