#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float scale_factor,
    cudaStream_t stream);

extern "C" void custom_relu_launcher(
    const float* input,
    float* output,
    size_t numel,
    cudaStream_t stream);

// Helper to check CUDA tensor dtype/device
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// custom_upsample_bilinear2d binding
// Input: (N, C, H, W) float32 CUDA tensor, scale_factor float
// Output: (N, C, floor(H*scale_factor), floor(W*scale_factor)) float32 CUDA tensor
torch::Tensor custom_upsample_bilinear2d(torch::Tensor input, double scale_factor) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    // If input is not contiguous, make it contiguous for correct indexing in the kernel.
    // Note: This may affect performance but ensures correctness for arbitrary strides.
    torch::Tensor input_contig = input.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    // Compute output dimensions as in PyTorch: floor(H * scale_factor)
    int outH = static_cast<int>(std::floor(H * scale_factor));
    int outW = static_cast<int>(std::floor(W * scale_factor));
    TORCH_CHECK(outH > 0 && outW > 0, "scale_factor too small, output size must be positive");

    // Allocate output tensor (contiguous, same dtype/device)
    auto output = torch::empty({N, C, outH, outW}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_upsample_bilinear2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(scale_factor),
        stream
    );

    return output;
}

// custom_relu binding
// Input: arbitrary shape float32 CUDA tensor, Output: same shape, relu applied
torch::Tensor custom_relu(torch::Tensor input) {
    check_cuda_float32(input, "input");

    // For arbitrary strides, make contiguous for simple elementwise kernel.
    // This ensures correctness; can be optimized later with a stride-aware kernel if needed.
    torch::Tensor input_contig = input.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    size_t numel = input_contig.numel();

    custom_relu_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_bilinear2d", &custom_upsample_bilinear2d,
          "Bilinear upsample (align_corners=False)",
          py::arg("input"), py::arg("scale_factor"));
    m.def("custom_relu", &custom_relu,
          "ReLU activation",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
