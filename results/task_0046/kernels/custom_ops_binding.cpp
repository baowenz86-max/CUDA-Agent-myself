#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void interpolate_bilinear_scale_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    float scale_factor,
    cudaStream_t stream);

extern "C" void relu_where_launcher(
    const float* input,
    float* output,
    int64_t total,
    cudaStream_t stream);

torch::Tensor interpolate_bilinear_scale(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "interpolate_bilinear_scale: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "interpolate_bilinear_scale: input must be float32");
    TORCH_CHECK(x.dim() == 4, "interpolate_bilinear_scale: input must be 4D (N,C,H,W)");
    TORCH_CHECK(scale_factor > 0, "interpolate_bilinear_scale: scale_factor must be positive");

    // Ensure input is contiguous for simpler launcher (kernel handles strides but we can avoid complexity)
    // Actually kernel supports arbitrary strides, so we can pass original strides. But ensure x is contiguous
    // if possible, or use .contiguous() only if needed.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = static_cast<int64_t>(std::floor(H_in * scale_factor));
    int64_t W_out = static_cast<int64_t>(std::floor(W_in * scale_factor));

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get strides after possible contiguous conversion
    int64_t stride_n_in = x.stride(0);
    int64_t stride_c_in = x.stride(1);
    int64_t stride_h_in = x.stride(2);
    int64_t stride_w_in = x.stride(3);
    int64_t stride_n_out = output.stride(0);
    int64_t stride_c_out = output.stride(1);
    int64_t stride_h_out = output.stride(2);
    int64_t stride_w_out = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    interpolate_bilinear_scale_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out,
        static_cast<float>(scale_factor),
        stream
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "interpolate_bilinear_scale: kernel launch failed");
    return output;
}

torch::Tensor relu_where(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_where: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_where: input must be float32");
    // relu_where_launcher assumes contiguous flat memory; make contiguous if not
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    int64_t total = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_where_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_where: kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("interpolate_bilinear_scale", &interpolate_bilinear_scale,
          "Bilinear upsampling with scale_factor (align_corners=False)",
          pybind11::arg("x"), pybind11::arg("scale_factor") = 1.0);
    m.def("relu_where", &relu_where,
          "Elementwise ReLU (max(x,0))",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
