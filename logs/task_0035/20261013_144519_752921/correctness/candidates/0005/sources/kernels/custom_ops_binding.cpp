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
    double scale_factor,
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
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    // Allow non-contiguous; we pass strides to launcher
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H_in = sizes[2];
    int W_in = sizes[3];
    if (H_in == 0 || W_in == 0) {
        // If input spatial dims are zero, output also zero in those dims.
        int H_out = static_cast<int>(std::floor(H_in * scale_factor));
        int W_out = static_cast<int>(std::floor(W_in * scale_factor));
        auto output = torch::empty({N, C, H_out, W_out}, x.options());
        return output;
    }
    // Compute output spatial dims: floor(H_in * scale_factor), floor(W_in * scale_factor)
    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output spatial dimensions must be positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get contiguous version of input if not already, but we can use original strides.
    // The launcher expects input pointer and strides. We'll make x contiguous to simplify? 
    // But that would copy data and lose performance. Since the input from model_new is typical contiguous,
    // we can check contiguity and use original if contiguous, else make contiguous.
    torch::Tensor input_contig = x.is_contiguous() ? x : x.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides of continuous input
    long long stride_n = input_contig.stride(0);
    long long stride_c = input_contig.stride(1);
    long long stride_h = input_contig.stride(2);
    long long stride_w = input_contig.stride(3);

    bilinear_upsample_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        scale_factor,  // pass double directly
        stride_n, stride_c, stride_h, stride_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in bilinear_upsample: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for logical_and_threshold
static torch::Tensor logical_and_threshold_forward(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    int total = x.numel();

    // Ensure input is contiguous; kernel reads linearly
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
