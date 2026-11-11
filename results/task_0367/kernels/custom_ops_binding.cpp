#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void custom_adaptive_avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_h, int out_w,
    cudaStream_t stream);

torch::Tensor custom_adaptive_avg_pool2d_forward(
    torch::Tensor x,
    int64_t output_size_h,
    int64_t output_size_w) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(output_size_h > 0 && output_size_w > 0, "Output size must be positive");

    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    int out_h = static_cast<int>(output_size_h);
    int out_w = static_cast<int>(output_size_w);

    auto output = torch::empty({N, C, out_h, out_w}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_adaptive_avg_pool2d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        out_h, out_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_adaptive_avg_pool2d(pybind11::module& m) {
    m.def("custom_adaptive_avg_pool2d_forward",
          &custom_adaptive_avg_pool2d_forward,
          "Adaptive avg pool2d forward",
          py::arg("x"), py::arg("output_size_h"), py::arg("output_size_w"));
}

REGISTER_BINDING(custom_adaptive_avg_pool2d, register_custom_adaptive_avg_pool2d);
