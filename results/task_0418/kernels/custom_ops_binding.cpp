#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void lp_pool2d_launcher_ext(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t kernel_size, int64_t stride,
    double norm_type,
    cudaStream_t stream
);

torch::Tensor lp_pool2d_forward(
    torch::Tensor input,
    double norm_type,
    int64_t kernel_size,
    c10::optional<int64_t> stride,
    bool ceil_mode
) {
    // Check device and dtype
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    // Resolve optional stride
    int64_t stride_val = stride.has_value() ? stride.value() : kernel_size;
    TORCH_CHECK(stride_val > 0, "stride must be positive");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");

    // Make contiguous if needed (launcher assumes contiguous NCHW)
    input = input.contiguous();

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Compute output spatial dimensions (ceil_mode False -> floor)
    int64_t outH, outW;
    if (ceil_mode) {
        outH = (H - kernel_size + stride_val - 1) / stride_val + 1;
        outW = (W - kernel_size + stride_val - 1) / stride_val + 1;
    } else {
        outH = (H - kernel_size) / stride_val + 1;
        outW = (W - kernel_size) / stride_val + 1;
    }

    TORCH_CHECK(outH > 0 && outW > 0, "output spatial dimensions must be positive");

    // Allocate output tensor on the same device
    auto output = torch::empty({N, C, outH, outW}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    lp_pool2d_launcher_ext(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kernel_size, stride_val,
        norm_type,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_lp_pool2d(pybind11::module& m) {
    m.def(
        "lp_pool2d_forward",
        &lp_pool2d_forward,
        "Lp pooling forward (CUDA)",
        py::arg("input"),
        py::arg("norm_type"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("ceil_mode")
    );
}

REGISTER_BINDING(lp_pool2d, register_lp_pool2d);
