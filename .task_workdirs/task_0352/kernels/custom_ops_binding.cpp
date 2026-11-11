// Binding for custom CUDA ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, double scale_factor,
    cudaStream_t stream);

extern "C" void reflection_pad2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

// Helper to check CUDA errors after launch
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Upsample bilinear 2D forward
torch::Tensor upsample_bilinear2d_forward(torch::Tensor input, double scale_factor) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");

    // Handle non-contiguous input by making contiguous (if needed)
    auto input_contig = input.contiguous();

    const int N = input_contig.size(0);
    const int C = input_contig.size(1);
    const int H_in = input_contig.size(2);
    const int W_in = input_contig.size(3);

    // Compute output size: floor(H * scale_factor), floor(W * scale_factor)
    const int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    const int W_out = static_cast<int>(std::floor(W_in * scale_factor));
    TORCH_CHECK(H_out > 0 && W_out > 0, "output size must be positive");

    // Allocate output tensor, contiguous
    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_bilinear2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out, scale_factor,
        stream
    );
    check_cuda_error("upsample_bilinear2d_launcher failed");

    return output;
}

// Reflection pad 2D forward
torch::Tensor reflection_pad2d_forward(torch::Tensor input, torch::Tensor padding) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    // Validate padding tensor: CPU int64, shape [4]
    TORCH_CHECK(padding.is_cpu(), "padding must be a CPU tensor");
    TORCH_CHECK(padding.dtype() == torch::kInt64, "padding must be int64");
    TORCH_CHECK(padding.numel() == 4, "padding must have 4 elements");
    // Ensure padding is contiguous
    auto padding_contig = padding.contiguous();
    const int64_t* pad_ptr = padding_contig.data_ptr<int64_t>();
    const int pad_left = static_cast<int>(pad_ptr[0]);
    const int pad_right = static_cast<int>(pad_ptr[1]);
    const int pad_top = static_cast<int>(pad_ptr[2]);
    const int pad_bottom = static_cast<int>(pad_ptr[3]);

    // Validate padding values are non-negative
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "padding values must be non-negative");

    // Handle non-contiguous input by making contiguous (if needed)
    auto input_contig = input.contiguous();

    const int N = input_contig.size(0);
    const int C = input_contig.size(1);
    const int H_in = input_contig.size(2);
    const int W_in = input_contig.size(3);

    const int H_out = H_in + pad_top + pad_bottom;
    const int W_out = W_in + pad_left + pad_right;

    // Allocate output tensor, contiguous
    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom,
        stream
    );
    check_cuda_error("reflection_pad2d_launcher failed");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d_forward", &upsample_bilinear2d_forward,
          "Bilinear 2D upsample forward",
          py::arg("input"), py::arg("scale_factor"));
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2D forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
