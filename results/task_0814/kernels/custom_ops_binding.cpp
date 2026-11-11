#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void upsample_bilinear_scale2_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

extern "C" void constant_pad2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t left, int64_t right, int64_t top, int64_t bottom,
    double value,
    cudaStream_t stream);

// Wrapper for bilinear upsampling with scale factor 2
torch::Tensor upsample_bilinear_scale2(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "upsample_bilinear_scale2: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "upsample_bilinear_scale2: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "upsample_bilinear_scale2: input must be float32");
    TORCH_CHECK(x.dim() == 4, "upsample_bilinear_scale2: input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t outH = 2 * H;
    int64_t outW = 2 * W;

    auto output = torch::empty({N, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    upsample_bilinear_scale2_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "upsample_bilinear_scale2: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for constant 2D padding
torch::Tensor constant_pad2d(
    torch::Tensor x,
    int64_t left,
    int64_t right,
    int64_t top,
    int64_t bottom,
    double value) {

    TORCH_CHECK(x.is_cuda(), "constant_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "constant_pad2d: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "constant_pad2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "constant_pad2d: input must be 4D (N, C, H, W)");
    TORCH_CHECK(left >= 0 && right >= 0 && top >= 0 && bottom >= 0,
                "constant_pad2d: padding widths must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t outH = H + top + bottom;
    int64_t outW = W + left + right;

    auto output = torch::empty({N, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    constant_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        left, right, top, bottom,
        value,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad2d: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear_scale2", &upsample_bilinear_scale2,
          "Bilinear upsampling with scale factor 2",
          py::arg("x"));
    m.def("constant_pad2d", &constant_pad2d,
          "Constant 2D padding",
          py::arg("x"),
          py::arg("left"),
          py::arg("right"),
          py::arg("top"),
          py::arg("bottom"),
          py::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
