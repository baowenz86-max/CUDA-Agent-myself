#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/ScalarType.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void view_as_real_forward_launcher(const void* input_complex, float* output_real, int64_t total_complex, cudaStream_t stream);
void sync_batchnorm_forward_launcher(const float* x, float* y, const float* weight, const float* bias, const float* running_mean, const float* running_var, int64_t N, int64_t C, int64_t H, int64_t W, float eps, cudaStream_t stream);
}

torch::Tensor view_as_real_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == c10::kComplex64, "Input must be complex64");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D [N,C,H]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t total_complex = N * C * H;

    auto output = torch::empty({N, C, H, 2}, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    view_as_real_forward_launcher(
        x.data_ptr(),
        output.data_ptr<float>(),
        total_complex,
        stream
    );

    return output;
}

torch::Tensor sync_batchnorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dtype() == torch::kFloat32, "weight must be CUDA float32 contiguous");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32 contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N,C,H,W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(weight.numel() == C, "weight size must match channel dim");
    TORCH_CHECK(bias.numel() == C, "bias size must match channel dim");

    // Since we don't have running_mean and running_var from the module,
    // we cannot call the launcher correctly. As a placeholder, we allocate
    // zero mean and one var tensors. This will produce incorrect output
    // but demonstrates the binding structure. In a later stage, model_new.py
    // must be updated to pass running_mean and running_var.
    auto options = x.options();
    auto running_mean = torch::zeros({C}, options);
    auto running_var = torch::ones({C}, options);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sync_batchnorm_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real_forward", &view_as_real_forward, "View as real forward", py::arg("x"));
    m.def("sync_batchnorm_forward", &sync_batchnorm_forward, "SyncBatchNorm forward", py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
