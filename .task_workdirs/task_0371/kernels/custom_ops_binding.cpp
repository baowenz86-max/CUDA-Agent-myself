#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void cross_map_lrn_forward_launcher(
    const float* input_ptr,
    float* output_ptr,
    int N, int C, int H, int W,
    long long input_n_stride, long long input_c_stride, long long input_h_stride, long long input_w_stride,
    long long output_n_stride, long long output_c_stride, long long output_h_stride, long long output_w_stride,
    int size,
    double alpha, double beta, double k,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor cross_map_lrn_forward(
    torch::Tensor input,
    int64_t size,
    double alpha,
    double beta,
    double k) {
    // Validate device and dtype
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)");

    // Allocate output with same properties
    auto output = torch::empty_like(input);

    // Get dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Get strides (long long for launcher)
    long long input_n_stride = input.stride(0);
    long long input_c_stride = input.stride(1);
    long long input_h_stride = input.stride(2);
    long long input_w_stride = input.stride(3);
    long long output_n_stride = output.stride(0);
    long long output_c_stride = output.stride(1);
    long long output_h_stride = output.stride(2);
    long long output_w_stride = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cross_map_lrn_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        output_n_stride, output_c_stride, output_h_stride, output_w_stride,
        static_cast<int>(size),
        alpha, beta, k,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_cross_map_lrn(pybind11::module& m) {
    m.def("cross_map_lrn_forward", &cross_map_lrn_forward,
          "CrossMapLRN2d forward",
          py::arg("input"),
          py::arg("size"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("k"));
}

// Auto-register
REGISTER_BINDING(cross_map_lrn, register_cross_map_lrn);
