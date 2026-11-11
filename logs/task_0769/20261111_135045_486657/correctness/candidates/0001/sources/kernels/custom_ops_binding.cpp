#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void concat_forward_launcher(
    const float* const* inputs,
    int num_inputs,
    int64_t outer_size,
    int64_t inner_size,
    int64_t concat_dim_size,
    float* output,
    cudaStream_t stream);

extern "C" void softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t spatial,
    cudaStream_t stream);

torch::Tensor concat_forward(torch::Tensor inputs, int64_t dim) {
    // inputs is expected to be a list of tensors; pybind will pass as torch::Tensor if we type it that way? 
    // Actually, we need to accept a list. But the contract fragment says "torch::Tensor inputs, int64_t dim".
    // However, the reference passes a list of tensors. We need to reflect that in binding.
    // The contract fragment may be wrong; we need to accept std::vector<torch::Tensor>.
    // Let's check model_new.py: cuda_extension.concat_forward(x, dim=1) where x is a list.
    // So we must use std::vector<torch.Tensor> or py::list and convert.
    // The contract fragment from binding_fragments likely generated incorrectly because it treats 'tensor' kind as single tensor, but the parameter shape description indicates a list.
    // We should implement according to actual usage: function takes a list of tensors and dim.
    // We'll define concat_forward(const std::vector<torch::Tensor>& inputs, int64_t dim) and register accordingly.
    // But the contract validation may expect parameters as per contract: tensor kind with shape "list of tensors". The contract may be manually inspected but the actual function signature in the binding will be from our C++ code.
    // We'll use std::vector<torch::Tensor>.
    TORCH_CHECK(!inputs.empty(), "concat_forward expects non-empty list");
    TORCH_CHECK(dim == 1, "concat_forward currently only supports dim=1");
    auto first = inputs[0];
    TORCH_CHECK(first.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(first.scalar_type() == torch::kFloat32, "All inputs must be float32");
    TORCH_CHECK(first.is_contiguous(), "All inputs must be contiguous");
    auto batch = first.size(0);
    auto channels = first.size(1);
    auto spatial = first.numel() / (batch * channels); // height*width
    int64_t num_inputs = inputs.size();
    // Validate all inputs same shape except dim=1 (channels may differ but here same)
    for (const auto& t : inputs) {
        TORCH_CHECK(t.is_cuda(), "All inputs must be CUDA tensors");
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "All inputs must be float32");
        TORCH_CHECK(t.is_contiguous(), "All inputs must be contiguous");
        TORCH_CHECK(t.dim() == 4, "Expected 4D tensors");
        TORCH_CHECK(t.size(0) == batch, "Batch mismatch");
        TORCH_CHECK(t.size(1) == channels, "Channel mismatch per tensor, expected same channels");
        TORCH_CHECK(t.size(2) * t.size(3) == spatial, "Spatial size mismatch");
    }
    auto output = torch::empty({batch, num_inputs * channels, first.size(2), first.size(3)}, first.options());
    // Prepare device pointer array
    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(num_inputs);
    for (const auto& t : inputs) {
        input_ptrs.push_back(t.data_ptr<float>());
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    concat_forward_launcher(
        input_ptrs.data(),
        static_cast<int>(num_inputs),
        batch,
        spatial,
        channels,
        output.data_ptr<float>(),
        stream);
    return output;
}

torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim == 1, "softmax_forward currently only supports dim=1");
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto spatial = input.numel() / (batch * channels);
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        spatial,
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("concat_forward", &concat_forward, "Concatenate list of tensors along dim=1", py::arg("inputs"), py::arg("dim"));
    m.def("softmax_forward", &softmax_forward, "Softmax along dim=1", py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
