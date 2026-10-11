#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void reflection_pad_histc_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream);

torch::Tensor reflection_pad_histc_forward(
    torch::Tensor input,
    int64_t padding,
    double start,
    double end,
    int64_t bins)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(bins > 0, "bins must be positive");
    TORCH_CHECK(start < end, "start must be less than end");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));
    int pad = static_cast<int>(padding);

    TORCH_CHECK(pad < H && pad < W, "padding size must be less than input dimensions");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad_histc_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        pad,
        static_cast<int>(bins),
        static_cast<float>(start),
        static_cast<float>(end),
        stream
    );

    return output;
}

void register_reflection_pad_histc(pybind11::module& m) {
    m.def("reflection_pad_histc_forward", &reflection_pad_histc_forward,
          "Fused reflection padding and histogram",
          py::arg("input"),
          py::arg("padding"),
          py::arg("start"),
          py::arg("end"),
          py::arg("bins"));
}

REGISTER_BINDING(reflection_pad_histc, register_reflection_pad_histc);