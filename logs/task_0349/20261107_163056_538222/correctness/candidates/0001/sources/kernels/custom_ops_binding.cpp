#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int B, int C, int H, int W,
                                   cudaStream_t stream);

extern "C" void hardswish_launcher(const float* input,
                                    float* output,
                                    long long total,
                                    cudaStream_t stream);

// Helper to check CUDA errors after launch
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// mean over dimension 0
static torch::Tensor mean_dim0(torch::Tensor x) {
    // Input checks: CUDA, float32, 4D
    TORCH_CHECK(x.is_cuda(), "mean_dim0: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim0: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_dim0: input must be 4D [B,C,H,W]");

    // The kernel currently assumes contiguous input; make contiguous
    auto x_contig = x.contiguous();

    int B = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "mean_dim0: non-positive dimensions");

    // Output shape [C,H,W]
    auto output = torch::empty({C, H, W}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim0_launcher(x_contig.data_ptr<float>(),
                       output.data_ptr<float>(),
                       B, C, H, W,
                       stream);
    check_cuda_error("mean_dim0 launch failed");

    return output;
}

// hardswish elementwise
static torch::Tensor hardswish(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardswish: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardswish: input must be float32");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    long long total = x_contig.numel();
    if (total > 0) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        hardswish_launcher(x_contig.data_ptr<float>(),
                           output.data_ptr<float>(),
                           total,
                           stream);
        check_cuda_error("hardswish launch failed");
    }

    return output;
}

// Registration
static void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim0", &mean_dim0, "Mean over dimension 0 (float32 CUDA)",
          pybind11::arg("x"));
    m.def("hardswish", &hardswish, "Hardswish activation (float32 CUDA)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
