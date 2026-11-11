#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

extern "C" void clamp_min_forward_launcher(
    const float* input,
    float* output,
    int total,
    float min_value,
    cudaStream_t stream);

// Wrapper for reflection_pad2d_forward
torch::Tensor reflection_pad2d_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Ensure contiguous, no channels_last support for simplicity in this kernel
    auto input_contig = input.contiguous();

    int N = static_cast<int>(input_contig.size(0));
    int C = static_cast<int>(input_contig.size(1));
    int H = static_cast<int>(input_contig.size(2));
    int W = static_cast<int>(input_contig.size(3));
    int Hout = H + pad_top + pad_bottom;
    int Wout = W + pad_left + pad_right;

    torch::Tensor output = torch::empty(
        {N, C, Hout, Wout}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(pad_left), static_cast<int>(pad_right),
        static_cast<int>(pad_top), static_cast<int>(pad_bottom),
        stream);

    return output;
}

// Wrapper for clamp_min_forward
torch::Tensor clamp_min_forward(
    torch::Tensor input,
    double min_value)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto input_contig = input.contiguous();
    int total = static_cast<int>(input_contig.numel());

    torch::Tensor output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_min_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        static_cast<float>(min_value),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          py::arg("input"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));

    m.def("clamp_min_forward", &clamp_min_forward,
          "Clamp min forward (CUDA)",
          py::arg("input"),
          py::arg("min_value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
