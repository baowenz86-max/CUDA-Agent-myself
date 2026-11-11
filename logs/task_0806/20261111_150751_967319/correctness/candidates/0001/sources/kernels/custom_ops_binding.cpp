#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void stack_tensors_launcher(
    const float* in0,
    const float* in1,
    const float* in2,
    const float* in3,
    float* out,
    int inner_numel,
    cudaStream_t stream);

extern "C" void softmax_dim0_launcher(
    const float* x,
    float* y,
    int inner_numel,
    cudaStream_t stream);

namespace {

torch::Tensor stack_tensors_forward(std::vector<torch::Tensor> input_list, int64_t dim) {
    // Validate dim = 0 for this task
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for stack_tensors");
    TORCH_CHECK(input_list.size() == 4, "input_list must contain exactly 4 tensors");

    // Validate each tensor
    for (size_t i = 0; i < input_list.size(); ++i) {
        const auto& t = input_list[i];
        TORCH_CHECK(t.is_cpu(), "All input tensors must be on CPU");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All input tensors must be float32");
        TORCH_CHECK(t.is_contiguous(), "All input tensors must be contiguous");
        TORCH_CHECK(t.sizes() == std::vector<int64_t>({512, 3, 128, 128}), "Each tensor must have shape [512,3,128,128]");
    }

    // Define dimensions
    int64_t inner_numel = 512 * 3 * 128 * 128;
    std::vector<int64_t> out_shape = {4, 512, 3, 128, 128};

    // Allocate output tensor on CUDA
    auto out = torch::empty(out_shape, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Move input tensors to CUDA (if not already) - they should be CPU, so transfer
    std::vector<torch::Tensor> cuda_inputs;
    cuda_inputs.reserve(4);
    for (auto& t : input_list) {
        cuda_inputs.push_back(t.to(torch::kCUDA, /*non_blocking=*/false));
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointers
    stack_tensors_launcher(
        cuda_inputs[0].data_ptr<float>(),
        cuda_inputs[1].data_ptr<float>(),
        cuda_inputs[2].data_ptr<float>(),
        cuda_inputs[3].data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(inner_numel),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "stack_tensors_launcher failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor softmax_dim0_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.sizes() == std::vector<int64_t>({4, 512, 3, 128, 128}), "Input must have shape [4,512,3,128,128]");

    int64_t inner_numel = 512 * 3 * 128 * 128;
    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim0_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(inner_numel),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_dim0_launcher failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("stack_tensors", &stack_tensors_forward, "Stack four tensors along dim 0 on CUDA",
          pybind11::arg("input_list"), pybind11::arg("dim"));
    m.def("softmax_dim0", &softmax_dim0_forward, "Softmax along dim 0",
          pybind11::arg("x"));
}

}  // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
