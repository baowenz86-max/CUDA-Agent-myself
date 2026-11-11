#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <climits>

// Declaration of launcher from custom_ops.cu
extern "C" void fused_tril_instance_norm_eval_forward(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    float eps,
    int diagonal,
    int N,
    int C,
    int H,
    int W,
    int stride_N,
    int stride_C,
    int stride_H,
    int stride_W,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Implementation of tril_forward: applies lower-triangular mask
torch::Tensor tril_forward(torch::Tensor x, int64_t diagonal) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input x must have at least 2 dimensions");

    // Allocate output same shape as x, contiguous
    auto output = torch::empty_like(x);

    // Dummy running stats for fused kernel (identity for normalization)
    int C = x.size(1); // assuming shape [N, C, H, W], but we generalize to last 2 dims
    // If x has more than 4 dims, we treat all leading dims as batch; but launcher expects N,C,H,W, so we assume 4D for now.
    // We can handle general by treating N = product of all dims except last two; C=1? But original operator expects [16,3,1024,1024] with C=3.
    // The contract says shape general N,C,H,W but current implementation assumes 4D. We will enforce 4D.
    TORCH_CHECK(x.dim() == 4, "tril_forward currently supports only 4D tensors [N,C,H,W]");
    int N = x.size(0);
    int C_dim = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate dummy running_mean and running_var of size C_dim on CUDA, set to 0 and 1 respectively
    auto running_mean = torch::zeros({C_dim}, x.options());
    auto running_var = torch::ones({C_dim}, x.options());

    cudaStream_t stream = get_current_stream();

    fused_tril_instance_norm_eval_forward(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        0.0f, // eps = 0, so rsqrt(1+0)=1, no scaling
        static_cast<int>(diagonal),
        N,
        C_dim,
        H,
        W,
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(2)),
        static_cast<int>(x.stride(3)),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_tril_instance_norm_eval_forward failed: ", cudaGetErrorString(err));

    return output;
}

// Implementation of instance_norm_eval_forward: applies per-channel normalization (no affine)
torch::Tensor instance_norm_eval_forward(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.dim() == 4, "instance_norm_eval_forward currently supports only 4D tensors [N,C,H,W]");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channel dimension");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channel dimension");

    // Allocate output same shape as x, contiguous
    auto output = torch::empty_like(x);

    cudaStream_t stream = get_current_stream();

    // Use INT_MIN for diagonal to always satisfy h >= w+INT_MIN -> true, so no masking
    fused_tril_instance_norm_eval_forward(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        INT_MIN,
        N,
        C,
        H,
        W,
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(2)),
        static_cast<int>(x.stride(3)),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_tril_instance_norm_eval_forward failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("tril_forward", &tril_forward, "Apply lower triangular mask",
          py::arg("x"), py::arg("diagonal"));
    m.def("instance_norm_eval_forward", &instance_norm_eval_forward, "Instance normalization eval mode",
          py::arg("x"), py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
