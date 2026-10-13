#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_round_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void custom_interpolate_launcher(const float* input, float* output,
                                            int N, int C, int H_in, int W_in,
                                            int H_out, int W_out, cudaStream_t stream);
extern "C" void custom_argmax_launcher(const float* input, int64_t* output,
                                       int N, int C, int H, int W, cudaStream_t stream);

namespace {

cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor custom_round_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_round: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_round: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_round: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_round: expected 4D input");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    custom_round_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        getCurrentStream()
    );
    return output;
}

torch::Tensor custom_interpolate_forward(torch::Tensor x, double scale_factor, int64_t mode) {
    TORCH_CHECK(x.is_cuda(), "custom_interpolate: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_interpolate: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_interpolate: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_interpolate: expected 4D input [N, C, H, W]");
    TORCH_CHECK(mode == 0, "custom_interpolate: only bilinear mode (0) is supported");
    TORCH_CHECK(scale_factor == 2.0, "custom_interpolate: only scale_factor=2.0 is supported");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int H_out = static_cast<int>(H_in * scale_factor);
    int W_out = static_cast<int>(W_in * scale_factor);

    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    custom_interpolate_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        getCurrentStream()
    );
    return output;
}

torch::Tensor custom_argmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_argmax: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_argmax: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_argmax: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_argmax: expected 4D input [N, C, H, W]");
    TORCH_CHECK(dim == 0, "custom_argmax: only dim=0 is supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({C, H, W}, x.options().dtype(torch::kInt64));
    custom_argmax_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        getCurrentStream()
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_round", &custom_round_forward, "Round input tensor element-wise",
          pybind11::arg("x"));
    m.def("custom_interpolate", &custom_interpolate_forward, "Bilinear interpolate with scale_factor=2.0",
          pybind11::arg("x"), pybind11::arg("scale_factor") = 2.0, pybind11::arg("mode") = 0);
    m.def("custom_argmax", &custom_argmax_forward, "Argmax along dim=0",
          pybind11::arg("x"), pybind11::arg("dim") = 0);
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
