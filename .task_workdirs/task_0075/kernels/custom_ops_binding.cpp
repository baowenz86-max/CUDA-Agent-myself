#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int pad,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    cudaStream_t stream
);

extern "C" void count_nonzero_forward_launcher(
    const float* input,
    long long total_elements,
    int64_t* output_count,
    cudaStream_t stream
);

extern "C" void upsample_bilinear2d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    float scale_factor,
    bool align_corners,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    cudaStream_t stream
);

// Helper: ensure input is CUDA float32 tensor
torch::Tensor check_cuda_float32(const torch::Tensor& input, const char* name) {
    TORCH_CHECK(input.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, name, " must be float32");
    return input.contiguous();
}

// 1. ReflectionPad2d
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Make contiguous if not already (reflection_pad2d launcher expects contiguous output, input strides can be handled but we'll pass contiguous input for simplicity)
    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);
    int64_t pad = padding;
    int64_t outH = H + 2 * pad;
    int64_t outW = W + 2 * pad;

    auto output = torch::empty({N, C, outH, outW}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad),
        static_cast<int>(input_contig.stride(0)),
        static_cast<int>(input_contig.stride(1)),
        static_cast<int>(input_contig.stride(2)),
        static_cast<int>(input_contig.stride(3)),
        stream
    );

    return output;
}

// 2. CountNonzero
torch::Tensor count_nonzero_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    // Make contiguous to simplify counting (launcher assumes contiguous float array)
    auto input_contig = input.contiguous();

    long long total_elements = static_cast<long long>(input_contig.numel());

    auto output = torch::zeros({}, input_contig.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    count_nonzero_forward_launcher(
        input_contig.data_ptr<float>(),
        total_elements,
        reinterpret_cast<int64_t*>(output.data_ptr<int64_t>()),
        stream
    );

    return output;
}

// 3. UpsampleBilinear2d
torch::Tensor upsample_bilinear2d_forward(torch::Tensor input, double scale_factor, bool align_corners) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");

    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t outH = static_cast<int64_t>(std::floor(H * scale_factor));
    int64_t outW = static_cast<int64_t>(std::floor(W * scale_factor));

    auto output = torch::empty({N, C, outH, outW}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_bilinear2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<float>(scale_factor),
        align_corners,
        static_cast<int>(input_contig.stride(0)),
        static_cast<int>(input_contig.stride(1)),
        static_cast<int>(input_contig.stride(2)),
        static_cast<int>(input_contig.stride(3)),
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("count_nonzero_forward", &count_nonzero_forward, "Count nonzero elements",
          py::arg("input"));
    m.def("upsample_bilinear2d_forward", &upsample_bilinear2d_forward, "Bilinear upsample forward",
          py::arg("input"), py::arg("scale_factor"), py::arg("align_corners") = false);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
