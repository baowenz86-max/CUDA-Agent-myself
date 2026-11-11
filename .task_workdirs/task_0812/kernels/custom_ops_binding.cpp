// PyTorch bindings for custom upsample and rot90 CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_upsample_bilinear2x_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void custom_rot90_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int k,
    cudaStream_t stream);

static torch::Tensor custom_upsample_bilinear2x(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_upsample_bilinear2x: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "custom_upsample_bilinear2x: input dtype must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_upsample_bilinear2x: input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.is_contiguous(), "custom_upsample_bilinear2x: input must be contiguous");

    const int N = static_cast<int>(input.size(0));
    const int C = static_cast<int>(input.size(1));
    const int H = static_cast<int>(input.size(2));
    const int W = static_cast<int>(input.size(3));
    const int outH = H * 2;
    const int outW = W * 2;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_upsample_bilinear2x_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_upsample_bilinear2x: CUDA error: ", cudaGetErrorString(err));

    return output;
}

static torch::Tensor custom_rot90(torch::Tensor input, int64_t k, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(input.is_cuda(), "custom_rot90: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "custom_rot90: input dtype must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_rot90: input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.is_contiguous(), "custom_rot90: input must be contiguous");
    TORCH_CHECK(dim0 == 2 && dim1 == 3, "custom_rot90: only dims=(2,3) supported");

    const int N = static_cast<int>(input.size(0));
    const int C = static_cast<int>(input.size(1));
    const int H = static_cast<int>(input.size(2));
    const int W = static_cast<int>(input.size(3));

    int k_mod = static_cast<int>(((k % 4) + 4) % 4);
    torch::Tensor output;
    if (k_mod == 1 || k_mod == 3) {
        // Rotated by 90 or 270 degrees: output shape [N, C, W, H]
        output = torch::empty({N, C, W, H}, input.options());
    } else {
        // k_mod == 0 or 2: shape unchanged
        output = torch::empty({N, C, H, W}, input.options());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_rot90_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        k_mod,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_rot90: CUDA error: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_bilinear2x", &custom_upsample_bilinear2x,
          "Bilinear 2x upsampling (align_corners=False)",
          py::arg("input"));
    m.def("custom_rot90", &custom_rot90,
          "Rotate last two dimensions by 90 degrees (k modulo 4)",
          py::arg("input"), py::arg("k"), py::arg("dim0"), py::arg("dim1"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);