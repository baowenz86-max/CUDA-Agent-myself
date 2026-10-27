#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int pad_size,
    cudaStream_t stream
);

extern "C" void maximum_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void minimum_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream
);

static torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t pad_size) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_h = input.size(2);
    int64_t in_w = input.size(3);
    int64_t out_h = in_h + 2 * pad_size;
    int64_t out_w = in_w + 2 * pad_size;

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        static_cast<int>(pad_size),
        stream
    );
    return output;
}

static torch::Tensor maximum_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Both inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Both inputs must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Both inputs must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input shapes must match for elementwise maximum");

    auto output = torch::empty_like(a);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maximum_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        a.numel(),
        stream
    );
    return output;
}

static torch::Tensor minimum_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Both inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Both inputs must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Both inputs must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input shapes must match for elementwise minimum");

    auto output = torch::empty_like(a);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    minimum_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        a.numel(),
        stream
    );
    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          pybind11::arg("input"), pybind11::arg("pad_size"));
    m.def("maximum_forward", &maximum_forward,
          "Elementwise maximum (CUDA)",
          pybind11::arg("a"), pybind11::arg("b"));
    m.def("minimum_forward", &minimum_forward,
          "Elementwise minimum (CUDA)",
          pybind11::arg("a"), pybind11::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
