#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_deg2rad_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void custom_reflection_pad2d_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int pad_t,
    int pad_l,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    cudaStream_t stream
);

// Forward wrapper for custom_deg2rad
torch::Tensor custom_deg2rad_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_deg2rad: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_deg2rad: input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_deg2rad_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Forward wrapper for custom_reflection_pad2d
torch::Tensor custom_reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_l,
    int64_t pad_r,
    int64_t pad_t,
    int64_t pad_b
) {
    TORCH_CHECK(x.is_cuda(), "custom_reflection_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_reflection_pad2d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_reflection_pad2d: input must be 3D (N, H, W)");
    TORCH_CHECK(pad_l >= 0 && pad_r >= 0 && pad_t >= 0 && pad_b >= 0,
                "custom_reflection_pad2d: padding must be non-negative");

    int64_t N = x.size(0);
    int64_t H_in = x.size(1);
    int64_t W_in = x.size(2);
    int64_t H_out = H_in + pad_t + pad_b;
    int64_t W_out = W_in + pad_l + pad_r;

    auto output = torch::empty({N, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_reflection_pad2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        1,          // C dimension not present, set to 1
        H_in,
        W_in,
        H_out,
        W_out,
        static_cast<int>(pad_t),
        static_cast<int>(pad_l),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        1,          // in_stride_w is 1 for the last dimension of 3D input
        // Actually need to pass in_stride_c, in_stride_h, in_stride_w properly
        // Since input is 3D, C=1, stride_c is not used; we pass x.stride(1) as in_stride_h,
        // and x.stride(2) as in_stride_w; in_stride_n = x.stride(0); in_stride_c = 0 (unused)
        0,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_deg2rad", &custom_deg2rad_forward,
          "Convert angles from degrees to radians",
          py::arg("x"));
    m.def("custom_reflection_pad2d", &custom_reflection_pad2d_forward,
          "Reflection pad 2D",
          py::arg("x"),
          py::arg("pad_l"),
          py::arg("pad_r"),
          py::arg("pad_t"),
          py::arg("pad_b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
