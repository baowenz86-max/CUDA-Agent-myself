#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    const int64_t* y_strides,
    cudaStream_t stream
);

extern "C" void max_reduce_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    cudaStream_t stream
);

torch::Tensor batch_norm_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "batch_norm_inference: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "batch_norm_inference: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "batch_norm_inference: bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "batch_norm_inference: running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "batch_norm_inference: running_var must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "batch_norm_inference: x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "batch_norm_inference: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "batch_norm_inference: bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "batch_norm_inference: running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "batch_norm_inference: running_var must be float32");

    TORCH_CHECK(x.dim() == 4, "batch_norm_inference: x must be 4D (N,C,H,W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(weight.numel() == C, "batch_norm_inference: weight size mismatch");
    TORCH_CHECK(bias.numel() == C, "batch_norm_inference: bias size mismatch");
    TORCH_CHECK(running_mean.numel() == C, "batch_norm_inference: running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == C, "batch_norm_inference: running_var size mismatch");

    // Allocate output tensor: same shape as input, contiguous
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Prepare strides arrays on host (these are int64_t)
    int64_t x_strides[4] = {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
    int64_t y_strides[4] = {y.stride(0), y.stride(1), y.stride(2), y.stride(3)};

    // Copy strides to device? The launcher expects pointer to const int64_t* (device pointer).
    // The .cu kernel dereferences x_strides and y_strides in device code.
    // Therefore we need to allocate device memory for strides and copy them.
    // But this would introduce a cudaMalloc/cudaMemcpy overhead per call.
    // Alternative: pass the strides as kernel arguments by value (within the launcher) and modify the launcher,
    // but we cannot modify .cu now. However, the launcher signature currently takes pointers to int64_t arrays.
    // In the original .cu, the launcher just passes those pointers directly to the kernel without dereferencing on host.
    // So they must be device pointers. We need to allocate device memory for stride arrays and copy host values.
    // To avoid per-call allocation overhead, we could use a static cache, but simplest is to allocate small buffers each time.
    // This is acceptable for correctness; performance can be improved later by modifying .cu.

    int64_t* d_x_strides = nullptr;
    int64_t* d_y_strides = nullptr;
    cudaMalloc(&d_x_strides, 4 * sizeof(int64_t));
    cudaMalloc(&d_y_strides, 4 * sizeof(int64_t));
    cudaMemcpyAsync(d_x_strides, x_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_strides, y_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Call launcher
    batch_norm_inference_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        d_x_strides,
        d_y_strides,
        stream
    );

    // Free device stride buffers after kernel completes (stream ordered)
    cudaFreeAsync(d_x_strides, stream);
    cudaFreeAsync(d_y_strides, stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm_inference kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

torch::Tensor max_reduce(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "max_reduce: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_reduce: x must be float32");
    TORCH_CHECK(x.dim() == 4, "max_reduce: x must be 4D (N,C,H,W)");
    TORCH_CHECK(dim == 1, "max_reduce: only dim=1 is supported");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Ensure input is contiguous; the .cu kernel expects x_strides to be provided,
    // but the reduction kernel assumes contiguous access pattern? Actually it uses x_strides,
    // so it can handle strides. But to be safe and match the expected output, we make input contiguous
    // if needed. However, batch_norm_inference returns contiguous, so x is contiguous here.
    x = x.contiguous();

    // Allocate output tensor: [N, H, W] contiguous, float32
    auto out = torch::empty({N, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Prepare strides array on host (input is contiguous, so strides are [C*H*W, H*W, W, 1])
    int64_t x_strides[4] = {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
    int64_t* d_x_strides = nullptr;
    cudaMalloc(&d_x_strides, 4 * sizeof(int64_t));
    cudaMemcpyAsync(d_x_strides, x_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Call launcher
    max_reduce_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        d_x_strides,
        stream
    );

    cudaFreeAsync(d_x_strides, stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_reduce kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_inference", &batch_norm_inference,
          "BatchNorm inference forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("max_reduce", &max_reduce,
          "Max reduce along dim",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
