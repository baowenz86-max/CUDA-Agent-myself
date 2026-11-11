#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void launch_linear(const float* input, const float* weight, const float* bias,
                              float* output, int64_t batch_size, int64_t in_features,
                              int64_t out_features, cudaStream_t stream);

extern "C" void launch_gelu(const float* input, float* output, int64_t numel, cudaStream_t stream);

extern "C" void launch_gt(const float* input, bool* output, float threshold, int64_t numel,
                         cudaStream_t stream);

extern "C" void launch_add(const bool* input, const float* other, float* output,
                          int64_t batch_size, int64_t in_features, cudaStream_t stream);

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight must have same in_features");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias_tensor.size(0) == weight.size(0), "bias must have out_features elements");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_linear(input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
                  output.data_ptr<float>(), batch_size, in_features, out_features, stream);

    return output;
}

torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_gelu(input.data_ptr<float>(), output.data_ptr<float>(), numel, stream);

    return output;
}

torch::Tensor gt_forward(torch::Tensor input, double threshold) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    int64_t numel = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_gt(input.data_ptr<float>(), output.data_ptr<bool>(), static_cast<float>(threshold),
              numel, stream);

    return output;
}

torch::Tensor add_forward(torch::Tensor input, torch::Tensor other) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(other.is_cuda(), "other must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBool, "input must be bool");
    TORCH_CHECK(other.dtype() == torch::kFloat32, "other must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(other.dim() == 3, "other must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(other.is_contiguous(), "other must be contiguous");

    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    TORCH_CHECK(other.size(0) == in_features, "other must have first dim == in_features");
    TORCH_CHECK(other.size(1) == 1 && other.size(2) == 1, "other must have shape [in_features, 1, 1]");

    auto output = torch::empty({in_features, batch_size, in_features}, other.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_add(input.data_ptr<bool>(), other.data_ptr<float>(), output.data_ptr<float>(),
               batch_size, in_features, stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear transformation: output = input @ weight.T + bias",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none());
    m.def("gelu_forward", &gelu_forward,
          "GELU activation (exact erf)",
          pybind11::arg("input"));
    m.def("gt_forward", &gt_forward,
          "Element-wise greater-than",
          pybind11::arg("input"), pybind11::arg("threshold") = 0.5);
    m.def("add_forward", &add_forward,
          "Broadcasting addition between bool and float",
          pybind11::arg("input"), pybind11::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);