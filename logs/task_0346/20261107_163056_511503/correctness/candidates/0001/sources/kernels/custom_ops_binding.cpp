#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    int scale_factor,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

// PyTorch wrapper for bilinear upsampling
torch::Tensor upsample_bilinear2d(torch::Tensor input, int64_t scale_factor) {
    // Validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(scale_factor >= 1, "scale_factor must be >= 1");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int sf = static_cast<int>(scale_factor);
    int outH = H * sf;
    int outW = W * sf;

    // Allocate output (contiguous)
    auto output = torch::empty({N, C, outH, outW}, input.options());

    // Ensure input is contiguous (or get strides in elements)
    auto input_contig = input.contiguous();
    auto output_contig = output.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides in elements (for contiguous tensors, but also handle non-contiguous correctly)
    // Use data_ptr directly with original strides if possible; since we call contiguous(),
    // we can rely on contiguous strides, but pass actual strides anyway.
    long long in_stride_n = input_contig.stride(0);
    long long in_stride_c = input_contig.stride(1);
    long long in_stride_h = input_contig.stride(2);
    long long in_stride_w = input_contig.stride(3);

    long long out_stride_n = output_contig.stride(0);
    long long out_stride_c = output_contig.stride(1);
    long long out_stride_h = output_contig.stride(2);
    long long out_stride_w = output_contig.stride(3);

    // Call launcher
    upsample_bilinear2d_launcher(
        input_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        sf,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    return output;
}

// Registration
void register_upsample_bilinear2d(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Bilinear upsampling (align_corners=False)",
          pybind11::arg("input"),
          pybind11::arg("scale_factor"));
}

REGISTER_BINDING(upsample_bilinear2d, register_upsample_bilinear2d);
