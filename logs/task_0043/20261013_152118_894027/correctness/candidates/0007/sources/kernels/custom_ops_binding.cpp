// Binding for view_as_real, permute_bhwc, clip custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void view_as_real_launcher(
    const void* input_complex,
    float* output_real,
    int64_t total_complex,
    cudaStream_t stream);

extern "C" void permute_bhwc_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t H, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_w, int64_t in_stride_c,
    cudaStream_t stream);

extern "C" void clip_launcher(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    int64_t B, int64_t H, int64_t C, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_c, int64_t in_stride_w,
    cudaStream_t stream);

// Helper to check CUDA error after kernel launch
static void check_cuda_error_after_launch() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// 1. view_as_real: complex64 [B,H,W] -> float32 [B,H,W,2]
torch::Tensor view_as_real_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "view_as_real: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kComplex64, "view_as_real: input dtype must be complex64");
    TORCH_CHECK(x.dim() == 3, "view_as_real: input must be 3D");

    // Ensure input is contiguous (kernel assumes contiguous complex layout)
    auto x_contig = x.contiguous();

    int64_t B = x_contig.size(0);
    int64_t H = x_contig.size(1);
    int64_t W = x_contig.size(2);
    int64_t total_complex = B * H * W;

    // Allocate output: shape [B, H, W, 2], float32, contiguous
    auto output = torch::empty({B, H, W, 2}, x_contig.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    view_as_real_launcher(
        x_contig.data_ptr<c10::complex<float>>(),
        output.data_ptr<float>(),
        total_complex,
        stream
    );

    check_cuda_error_after_launch();
    return output;
}

// 2. permute_bhwc: [B,H,W,2] -> [B,H,2,W]
torch::Tensor permute_bhwc_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "permute_bhwc: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "permute_bhwc: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "permute_bhwc: input must be 4D");
    TORCH_CHECK(x.size(3) == 2, "permute_bhwc: last dim must be 2");

    int64_t B = x.size(0);
    int64_t H = x.size(1);
    int64_t W = x.size(2);
    // C = 2 implied by size(3)

    // Get input strides in elements
    int64_t in_stride_b = x.stride(0);
    int64_t in_stride_h = x.stride(1);
    int64_t in_stride_w = x.stride(2);
    int64_t in_stride_c = x.stride(3);

    // Allocate output: shape [B, H, 2, W], contiguous
    auto output = torch::empty({B, H, 2, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    permute_bhwc_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, W,
        in_stride_b, in_stride_h, in_stride_w, in_stride_c,
        stream
    );

    check_cuda_error_after_launch();
    return output;
}

// 3. clip: elementwise clamp
torch::Tensor clip_forward(torch::Tensor x, double min_val, double max_val) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "clip: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "clip: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "clip: input must be 4D");

    int64_t B = x.size(0);
    int64_t H = x.size(1);
    int64_t C = x.size(2);
    int64_t W = x.size(3);

    // Get input strides in elements
    int64_t in_stride_b = x.stride(0);
    int64_t in_stride_h = x.stride(1);
    int64_t in_stride_c = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    // Allocate output: same shape, contiguous
    auto output = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clip_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        B, H, C, W,
        in_stride_b, in_stride_h, in_stride_c, in_stride_w,
        stream
    );

    check_cuda_error_after_launch();
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real", &view_as_real_forward, "Convert complex64 tensor to real float32 with extra last dim 2",
          py::arg("x"));
    m.def("permute_bhwc", &permute_bhwc_forward, "Permute [B,H,W,2] to [B,H,2,W]",
          py::arg("x"));
    m.def("clip", &clip_forward, "Elementwise clamp",
          py::arg("x"), py::arg("min") = 0.0, py::arg("max") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
