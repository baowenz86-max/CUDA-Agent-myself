#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

extern "C" void leaky_relu_forward_launcher(const float* input,
                                            float* output,
                                            float negative_slope,
                                            int64_t numel,
                                            cudaStream_t stream);

extern "C" void batch_norm3d_inference_launcher(const float* input,
                                                float* output,
                                                const float* weight,
                                                const float* bias,
                                                const float* running_mean,
                                                const float* running_var,
                                                float eps,
                                                int64_t N,
                                                int64_t C,
                                                int64_t spatial_size,
                                                cudaStream_t stream);

// Helper to check common tensor requirements
void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// tanh_forward
at::Tensor tanh_forward(const at::Tensor& input) {
    check_cuda_float32_contiguous(input, "input");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tanh_forward_launcher(input.data_ptr<float>(),
                          output.data_ptr<float>(),
                          input.numel(),
                          stream);
    return output;
}

// leaky_relu_forward
at::Tensor leaky_relu_forward(const at::Tensor& input, double negative_slope) {
    check_cuda_float32_contiguous(input, "input");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_forward_launcher(input.data_ptr<float>(),
                                output.data_ptr<float>(),
                                static_cast<float>(negative_slope),
                                input.numel(),
                                stream);
    return output;
}

// batch_norm3d_inference_forward
at::Tensor batch_norm3d_inference_forward(const at::Tensor& input,
                                          const at::Tensor& weight,
                                          const at::Tensor& bias,
                                          const at::Tensor& running_mean,
                                          const at::Tensor& running_var,
                                          double eps) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(weight, "weight");
    check_cuda_float32_contiguous(bias, "bias");
    check_cuda_float32_contiguous(running_mean, "running_mean");
    check_cuda_float32_contiguous(running_var, "running_var");
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N, C, D, H, W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);
    int64_t spatial_size = D * H * W;
    TORCH_CHECK(weight.numel() == C, "weight size mismatch");
    TORCH_CHECK(bias.numel() == C, "bias size mismatch");
    TORCH_CHECK(running_mean.numel() == C, "running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == C, "running_var size mismatch");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm3d_inference_launcher(input.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    weight.data_ptr<float>(),
                                    bias.data_ptr<float>(),
                                    running_mean.data_ptr<float>(),
                                    running_var.data_ptr<float>(),
                                    static_cast<float>(eps),
                                    N,
                                    C,
                                    spatial_size,
                                    stream);
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("tanh_forward", &tanh_forward, "Tanh forward");
    m.def("leaky_relu_forward", &leaky_relu_forward, "LeakyReLU forward",
          pybind11::arg("input"), pybind11::arg("negative_slope"));
    m.def("batch_norm3d_inference_forward", &batch_norm3d_inference_forward,
          "BatchNorm3d inference forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
