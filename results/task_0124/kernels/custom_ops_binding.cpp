#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(float* output, const float* input, int64_t numel, float min_val, float max_val, cudaStream_t stream);
extern "C" void pow_forward_launcher(float* output, const float* input, int64_t numel, float exponent, cudaStream_t stream);

// clamp_forward wrapper
torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure contiguous for linear kernel
    x = x.contiguous();
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(), numel, static_cast<float>(min_val), static_cast<float>(max_val), stream);
    return output;
}

// pow_forward wrapper
torch::Tensor pow_forward(torch::Tensor x, double exponent) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure contiguous for linear kernel
    x = x.contiguous();
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pow_forward_launcher(output.data_ptr<float>(), x.data_ptr<float>(), numel, static_cast<float>(exponent), stream);
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward, "Apply torch.clamp elementwise",
          pybind11::arg("x"), pybind11::arg("min_val"), pybind11::arg("max_val"));
    m.def("pow_forward", &pow_forward, "Apply torch.pow elementwise",
          pybind11::arg("x"), pybind11::arg("exponent"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);