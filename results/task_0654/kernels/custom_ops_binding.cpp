#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void copy_tensor_k_abs_out_launcher(float* output, const float* input,
                                                long long numel, cudaStream_t stream);
extern "C" void copy_tensor_k_argmin_out_launcher(const float* input, int64_t* output,
                                                   int B, int D, int H, int W,
                                                   long long in_stride_b, long long in_stride_d,
                                                   long long in_stride_h, long long in_stride_w,
                                                   long long out_stride_b, long long out_stride_h,
                                                   long long out_stride_w,
                                                   cudaStream_t stream);

// Wrapper for abs kernel
torch::Tensor copy_tensor_k_abs_out(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    long long numel = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    copy_tensor_k_abs_out_launcher(output.data_ptr<float>(), input.data_ptr<float>(), numel, stream);

    return output;
}

// Wrapper for argmin kernel
// Handles arbitrary dim, but currently only dim=1 is expected and optimized.
torch::Tensor copy_tensor_k_argmin_out(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    // Normalize negative dim
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");

    // For this specific task, dim=1 and input is 4D, but we can keep general logic for robustness.
    // Compute output sizes: remove the dim dimension
    auto sizes = input.sizes().vec();
    sizes.erase(sizes.begin() + dim);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
    auto output = torch::empty(sizes, options);

    // Currently the .cu launcher is specialized for dim=1 with 4D tensor.
    // For general dim, we need a different approach, but the contract only requires dim=1.
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for this kernel");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D for dim=1 argmin");

    int B = input.size(0);
    int D = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    long long in_stride_b = input.stride(0);
    long long in_stride_d = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);

    long long out_stride_b = output.stride(0);
    long long out_stride_h = output.stride(1);
    long long out_stride_w = output.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    copy_tensor_k_argmin_out_launcher(input.data_ptr<float>(), output.data_ptr<int64_t>(),
                                      B, D, H, W,
                                      in_stride_b, in_stride_d, in_stride_h, in_stride_w,
                                      out_stride_b, out_stride_h, out_stride_w,
                                      stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("copy_tensor_k_abs_out", &copy_tensor_k_abs_out, "Absolute value forward",
          py::arg("input"));
    m.def("copy_tensor_k_argmin_out", &copy_tensor_k_argmin_out, "Argmin forward",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
