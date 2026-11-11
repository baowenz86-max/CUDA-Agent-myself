#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void custom_rot90_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream);

torch::Tensor custom_rot90(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output shape: [N, H, C, W]
    auto output = torch::empty({N, H, C, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get input strides (in elements, not bytes)
    auto strides = x.strides();
    int64_t in_stride_n = strides[0];
    int64_t in_stride_c = strides[1];
    int64_t in_stride_h = strides[2];
    int64_t in_stride_w = strides[3];

    // Call launcher
    custom_rot90_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_rot90(pybind11::module& m) {
    m.def("custom_rot90", &custom_rot90,
          "Rotate input by 90 degrees clockwise in dims (1,2)",
          py::arg("x"));
}

REGISTER_BINDING(custom_rot90, register_custom_rot90);
