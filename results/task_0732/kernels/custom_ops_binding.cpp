#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void acos_stack_forward_launcher(const float* input,
                                             float* output,
                                             long long total,
                                             cudaStream_t stream);

extern "C" void cat_unbind_channels_forward_launcher(const float* input,
                                                      float* output,
                                                      int B, int C, int H, int W,
                                                      cudaStream_t stream);

// Wrapper for acos_stack: elementwise acos, output shape same as input
torch::Tensor acos_stack(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    long long total = x.numel();
    TORCH_CHECK(total > 0, "Input tensor must have at least one element");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    acos_stack_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "acos_stack kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for cat_unbind_channels: [B,C,H,W] -> [B,H,C*W]
torch::Tensor cat_unbind_channels(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "All dimensions must be positive");
    TORCH_CHECK(B <= INT_MAX && C <= INT_MAX && H <= INT_MAX && W <= INT_MAX,
                "Dimension exceeds int range");

    // Output shape [B, H, C*W]
    auto output = torch::empty({B, H, C * W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cat_unbind_channels_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cat_unbind_channels kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("acos_stack", &acos_stack,
          "Elementwise acos, output shape same as input",
          py::arg("x"));
    m.def("cat_unbind_channels", &cat_unbind_channels,
          "Permute [B,C,H,W] to [B,H,C*W]",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
