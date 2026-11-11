#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void upsample_bilinear2x_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream);

extern "C" void rot90_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int k,
    cudaStream_t stream);

// upsample_bilinear2x_forward: input (N,C,H,W) float32 CUDA -> output (N,C,2H,2W) float32 CUDA
torch::Tensor upsample_bilinear2x_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    // Output shape [N, C, 2H, 2W]
    auto output = torch::empty({N, C, 2*H, 2*W}, input.options());

    // Ensure output is contiguous (torch::empty is contiguous by default)
    TORCH_CHECK(output.is_contiguous(), "internal error: output not contiguous");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_bilinear2x_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "upsample_bilinear2x_forward launcher failed");
    return output;
}

// rot90_forward: input (N,C,H,W) float32 CUDA, k int, dims and dims2 specify axes (expected 2,3)
// output shape depends on k%4: for k=1 or 3, (N,C,W,H); for k=0 or 2, (N,C,H,W)
torch::Tensor rot90_forward(torch::Tensor input, int64_t k, int64_t dims, int64_t dims2) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(dims == 2 && dims2 == 3, "dims must be (2,3) for this operator");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    int kmod = ((k % 4) + 4) % 4;
    int64_t outH, outW;
    if (kmod == 0 || kmod == 2) {
        outH = H;
        outW = W;
    } else {
        outH = W;
        outW = H;
    }

    auto output = torch::empty({N, C, outH, outW}, input.options());
    TORCH_CHECK(output.is_contiguous(), "internal error: output not contiguous");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    rot90_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        static_cast<int>(k),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rot90_forward launcher failed");
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2x_forward", &upsample_bilinear2x_forward,
          "Bilinear upsampling with scale_factor=2",
          py::arg("input"));
    m.def("rot90_forward", &rot90_forward,
          "Rotate tensor 90 degrees in the last two dimensions",
          py::arg("input"), py::arg("k"), py::arg("dims"), py::arg("dims2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
