// PyTorch binding for custom unfold operations
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream);

extern "C" void unfold_amax_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream);

extern "C" void unfold_count_nonzero_launcher(
    const float* input,
    int64_t* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream);

// Helper to validate common parameters
static void validate_input(const torch::Tensor& input,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t dilation_h, int64_t dilation_w,
                           int64_t padding_h, int64_t padding_w) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
}

// unfold_forward: returns [N, C*kernel_h*kernel_w, L]
torch::Tensor unfold_forward(torch::Tensor input,
                             int64_t kernel_h, int64_t kernel_w,
                             int64_t dilation_h, int64_t dilation_w,
                             int64_t padding_h, int64_t padding_w) {
    validate_input(input, kernel_h, kernel_w, dilation_h, dilation_w, padding_h, padding_w);

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Compute output dimensions (L = out_h * out_w)
    int64_t out_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) + 1;
    int64_t out_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) + 1;
    int64_t L = out_h * out_w;
    int64_t Ckhw = C * kernel_h * kernel_w;

    auto output = torch::empty({N, Ckhw, L}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unfold_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        padding_h, padding_w,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "unfold_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// unfold_amax_forward: returns [N] float32
torch::Tensor unfold_amax_forward(torch::Tensor input,
                                  int64_t kernel_h, int64_t kernel_w,
                                  int64_t dilation_h, int64_t dilation_w,
                                  int64_t padding_h, int64_t padding_w) {
    validate_input(input, kernel_h, kernel_w, dilation_h, dilation_w, padding_h, padding_w);

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty({N}, input.options()); // float32

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unfold_amax_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        padding_h, padding_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "unfold_amax_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// unfold_count_nonzero_forward: returns [N] int64
torch::Tensor unfold_count_nonzero_forward(torch::Tensor input,
                                           int64_t kernel_h, int64_t kernel_w,
                                           int64_t dilation_h, int64_t dilation_w,
                                           int64_t padding_h, int64_t padding_w) {
    validate_input(input, kernel_h, kernel_w, dilation_h, dilation_w, padding_h, padding_w);

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty({N}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unfold_count_nonzero_launcher(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        padding_h, padding_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "unfold_count_nonzero_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("unfold_forward", &unfold_forward,
          "Unfold operation (debug)",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));

    m.def("unfold_amax_forward", &unfold_amax_forward,
          "Unfold then amax over dim=(1,2)",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));

    m.def("unfold_count_nonzero_forward", &unfold_count_nonzero_forward,
          "Unfold then count_nonzero over dim=(1,2)",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
