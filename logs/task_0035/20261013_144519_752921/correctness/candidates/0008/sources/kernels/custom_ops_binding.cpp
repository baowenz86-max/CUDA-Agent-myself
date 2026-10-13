// Binding for bilinear upsampling and logical_and_threshold
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void bilinear_upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream);

extern "C" void logical_and_threshold_launcher(
    const float* input,
    bool* output,
    int total,
    cudaStream_t stream);

// Wrapper for bilinear upsample
static torch::Tensor bilinear_upsample_forward(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H_in = sizes[2];
    int W_in = sizes[3];

    // Compute output spatial dims: floor(H_in * scale_factor), floor(W_in * scale_factor)
    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output spatial dimensions must be positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Ensure input is contiguous; launcher supports strides, but we can use contiguous to simplify.
    torch::Tensor input_contig = x.is_contiguous() ? x : x.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long stride_n = input_contig.stride(0);
    long long stride_c = input_contig.stride(1);
    long long stride_h = input_contig.stride(2);
    long long stride_w = input_contig.stride(3);

    bilinear_upsample_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        static_cast<float>(scale_factor),
        stride_n, stride_c, stride_h, stride_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in bilinear_upsample: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for logical_and_threshold
static torch::Tensor logical_and_threshold_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    int total = x.numel();

    torch::Tensor input_contig = x.is_contiguous() ? x : x.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    logical_and_threshold_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        total,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in logical_and_threshold: ", cudaGetErrorString(err));

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("bilinear_upsample", &bilinear_upsample_forward,
          "Bilinear upsampling (align_corners=False)",
          pybind11::arg("x"), pybind11::arg("scale_factor"));
    m.def("logical_and_threshold", &logical_and_threshold_forward,
          "Elementwise logical AND of (x > 0) and (x < 1)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
