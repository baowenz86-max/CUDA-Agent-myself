#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu (extern "C")
extern "C" void max_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    const long long* input_strides,
    int kD, int kH, int kW,
    int sD_stride, int sH_stride, int sW_stride,
    int pD, int pH, int pW,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream);

torch::Tensor max_pool3d_forward(
    torch::Tensor input,
    py::tuple kernel_size,
    py::tuple stride,
    py::tuple padding,
    int64_t dilation,
    bool ceil_mode) {

    TORCH_CHECK(input.is_cuda(), "max_pool3d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "max_pool3d_forward: input must be float32");
    TORCH_CHECK(input.dim() == 5, "max_pool3d_forward: input must be 5D (N,C,D,H,W)");

    TORCH_CHECK(kernel_size.size() == 3, "max_pool3d_forward: kernel_size must be a 3-tuple");
    TORCH_CHECK(stride.size() == 3, "max_pool3d_forward: stride must be a 3-tuple");
    TORCH_CHECK(padding.size() == 3, "max_pool3d_forward: padding must be a 3-tuple");

    // Convert tuple elements to int64_t
    int64_t kD = kernel_size[0].cast<int64_t>();
    int64_t kH = kernel_size[1].cast<int64_t>();
    int64_t kW = kernel_size[2].cast<int64_t>();
    int64_t sD = stride[0].cast<int64_t>();
    int64_t sH = stride[1].cast<int64_t>();
    int64_t sW = stride[2].cast<int64_t>();
    int64_t pD = padding[0].cast<int64_t>();
    int64_t pH = padding[1].cast<int64_t>();
    int64_t pW = padding[2].cast<int64_t>();

    // Validate positive, reasonable values
    TORCH_CHECK(kD > 0 && kH > 0 && kW > 0, "max_pool3d_forward: kernel sizes must be positive");
    TORCH_CHECK(sD > 0 && sH > 0 && sW > 0, "max_pool3d_forward: strides must be positive");
    TORCH_CHECK(pD >= 0 && pH >= 0 && pW >= 0, "max_pool3d_forward: paddings must be non-negative");
    TORCH_CHECK(dilation > 0, "max_pool3d_forward: dilation must be positive");

    // Get dimensions from input tensor
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    // Obtain input strides as long long array (element strides)
    std::vector<long long> input_strides(5);
    for (int i = 0; i < 5; ++i) {
        input_strides[i] = static_cast<long long>(input.stride(i));
    }

    // Compute output dimensions (matching PyTorch formula)
    auto compute_out_dim = [&](int64_t in, int64_t k, int64_t s, int64_t p, int64_t d, bool ceil) -> int64_t {
        int64_t numerator = in + 2 * p - d * (k - 1) - 1;
        if (ceil) {
            return (numerator + s - 1) / s + 1;
        } else {
            return numerator / s + 1;
        }
    };
    int64_t outD = compute_out_dim(D, kD, sD, pD, dilation, ceil_mode);
    int64_t outH = compute_out_dim(H, kH, sH, pH, dilation, ceil_mode);
    int64_t outW = compute_out_dim(W, kW, sW, pW, dilation, ceil_mode);

    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "max_pool3d_forward: invalid output dimensions");

    // Allocate output tensor of same dtype and device
    auto output = torch::empty({N, C, outD, outH, outW}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    max_pool3d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(D),
        static_cast<int>(H),
        static_cast<int>(W),
        input_strides.data(),
        static_cast<int>(kD),
        static_cast<int>(kH),
        static_cast<int>(kW),
        static_cast<int>(sD),
        static_cast<int>(sH),
        static_cast<int>(sW),
        static_cast<int>(pD),
        static_cast<int>(pH),
        static_cast<int>(pW),
        static_cast<int>(dilation),
        ceil_mode,
        stream
    );

    // Check for CUDA errors after launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "max_pool3d_forward: CUDA kernel launch failed");

    return output;
}

void register_max_pool3d(pybind11::module& m) {
    m.def("max_pool3d_forward", &max_pool3d_forward,
          "MaxPool3d forward (cuDNN-like)",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("ceil_mode"));
}

REGISTER_BINDING(max_pool3d, register_max_pool3d);
