#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void view_as_real_launcher(
    const float* input,
    float* output,
    long long d0, long long d1, long long d2,
    long long s0, long long s1, long long s2,
    long long os0, long long os1, long long os2, long long os3,
    cudaStream_t stream);

extern "C" void einsum_bchw_to_bhwc_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    cudaStream_t stream);

extern "C" void clip_launcher(
    const float* input,
    float* output,
    long long d0, long long d1, long long d2, long long d3,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    float min_val, float max_val,
    cudaStream_t stream);

// Helper to get current CUDA stream
static inline cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// 1. view_as_real: complex64 -> float32 with extra trailing dimension of size 2
torch::Tensor view_as_real_kernel(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "view_as_real_kernel: input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kComplex64, "view_as_real_kernel: input must be complex64");
    TORCH_CHECK(x.dim() == 3, "view_as_real_kernel: input must be 3D");

    // Input is [B, C, H] in complex elements; we need contiguous in terms of complex elements
    auto x_contig = x.contiguous(); // ensure strides are dense in complex elements

    long long B = x_contig.size(0);
    long long C = x_contig.size(1);
    long long H = x_contig.size(2);

    // Output shape: [B, C, H, 2]
    auto output = torch::empty({B, C, H, 2}, x_contig.options().dtype(torch::kFloat32));

    // For contiguous complex input, strides in complex elements: [C*H, H, 1]
    // Since data_ptr is float*, offsets must be in float units, so multiply by 2.
    long long s0 = C * H * 2LL;
    long long s1 = H * 2LL;
    long long s2 = 2LL;

    // Output strides from torch::empty: [C*H*2, H*2, 2, 1]
    long long os0 = output.stride(0);
    long long os1 = output.stride(1);
    long long os2 = output.stride(2);
    long long os3 = output.stride(3);

    cudaStream_t stream = getCurrentStream();

    view_as_real_launcher(
        reinterpret_cast<const float*>(x_contig.data_ptr<c10::complex<float>>()),
        output.data_ptr<float>(),
        B, C, H,
        s0, s1, s2,
        os0, os1, os2, os3,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "view_as_real_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// 2. einsum bchw->bhwc: transpose last two dimensions
torch::Tensor einsum_bchw_to_bhwc(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "einsum_bchw_to_bhwc: input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "einsum_bchw_to_bhwc: input must be float32");
    TORCH_CHECK(x.dim() == 4, "einsum_bchw_to_bhwc: input must be 4D");

    // Input shape: [B, C, H, W]
    long long B = x.size(0);
    long long C = x.size(1);
    long long H = x.size(2);
    long long W = x.size(3);

    // Output shape: [B, H, W, C] with reference stride [H*W*C, W*C, C, 1]
    auto output = torch::empty({B, H, W, C}, x.options());

    // Launcher supports arbitrary strides, but we must pass correct input strides.
    // We'll use input strides as provided (x may be non-contiguous).
    long long is0 = x.stride(0);
    long long is1 = x.stride(1);
    long long is2 = x.stride(2);
    long long is3 = x.stride(3);

    long long os0 = output.stride(0);
    long long os1 = output.stride(1);
    long long os2 = output.stride(2);
    long long os3 = output.stride(3);

    cudaStream_t stream = getCurrentStream();

    einsum_bchw_to_bhwc_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        is0, is1, is2, is3,
        os0, os1, os2, os3,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "einsum_bchw_to_bhwc_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// 3. clip: element-wise clamp with min_val and max_val
torch::Tensor clip_kernel(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "clip_kernel: input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "clip_kernel: input must be float32");
    TORCH_CHECK(x.dim() == 4, "clip_kernel: input must be 4D");

    // Input shape: [D0, D1, D2, D3]
    long long d0 = x.size(0);
    long long d1 = x.size(1);
    long long d2 = x.size(2);
    long long d3 = x.size(3);

    // Output tensor with same shape and strides as input
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    long long is0 = x.stride(0);
    long long is1 = x.stride(1);
    long long is2 = x.stride(2);
    long long is3 = x.stride(3);
    long long os0 = output.stride(0); // should equal is0
    long long os1 = output.stride(1);
    long long os2 = output.stride(2);
    long long os3 = output.stride(3);

    cudaStream_t stream = getCurrentStream();

    clip_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        d0, d1, d2, d3,
        is0, is1, is2, is3,
        os0, os1, os2, os3,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clip_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real_kernel", &view_as_real_kernel, "Convert complex64 tensor to float32 view with extra last dim of size 2", py::arg("x"));
    m.def("einsum_bchw_to_bhwc", &einsum_bchw_to_bhwc, "Einsum bchw->bhwc (transpose last two dims)", py::arg("x"));
    m.def("clip_kernel", &clip_kernel, "Clip tensor values to [min, max]", py::arg("x"), py::arg("min_val"), py::arg("max_val"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
