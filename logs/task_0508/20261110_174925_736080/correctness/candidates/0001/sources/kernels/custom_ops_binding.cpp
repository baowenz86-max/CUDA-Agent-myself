// Custom ops binding for stack and BCE loss
// This file wraps the launchers declared in custom_ops.cu

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

#include <vector>
#include <cstdint>

// Declare launchers from custom_ops.cu
extern "C" void stack_forward_launcher(const float* const* src_ptrs, float* dst, int64_t chunk_size, int64_t num_chunks, cudaStream_t stream);
extern "C" void bce_with_logits_loss_forward_launcher(const float* input, const float* target, float* output, int64_t n, cudaStream_t stream);

namespace {

// Helper: convert a Python iterable of tensors to vector<torch::Tensor>
std::vector<torch::Tensor> py_list_to_tensor_vector(const py::object& obj) {
    std::vector<torch::Tensor> tensors;
    // Try to cast to a list
    py::list lst = py::cast<py::list>(obj);
    for (auto item : lst) {
        tensors.push_back(py::cast<torch::Tensor>(item));
    }
    return tensors;
}

} // anonymous namespace

// stack_forward: takes a Python list of CPU tensors, moves to CUDA and stacks along new dim 0.
torch::Tensor stack_forward(const py::object& tensors_obj) {
    // Convert to vector of tensors
    auto tensors = py_list_to_tensor_vector(tensors_obj);
    TORCH_CHECK(!tensors.empty(), "stack_forward expects a non-empty list of tensors");
    
    // Validate: all tensors must be CPU float32 with same shape
    auto first = tensors[0];
    TORCH_CHECK(first.is_cpu(), "All input tensors must be on CPU");
    TORCH_CHECK(first.dtype() == torch::kFloat32, "All input tensors must be float32");
    TORCH_CHECK(first.is_contiguous(), "Input tensors must be contiguous");
    int64_t numel_per_tensor = first.numel();
    auto shape = first.sizes().vec();
    for (size_t i = 1; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].is_cpu(), "All input tensors must be on CPU");
        TORCH_CHECK(tensors[i].dtype() == torch::kFloat32, "All input tensors must be float32");
        TORCH_CHECK(tensors[i].is_contiguous(), "Input tensors must be contiguous");
        TORCH_CHECK(tensors[i].numel() == numel_per_tensor, "All tensors must have the same number of elements");
        TORCH_CHECK(tensors[i].sizes().vec() == shape, "All tensors must have the same shape");
    }

    // Move each tensor to CUDA if not already (they should be CPU, but we handle anyway)
    std::vector<torch::Tensor> cuda_tensors;
    cuda_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.is_cuda()) {
            cuda_tensors.push_back(t.contiguous());
        } else {
            cuda_tensors.push_back(t.to(torch::kCUDA).contiguous());
        }
    }

    int64_t num_chunks = static_cast<int64_t>(cuda_tensors.size());
    int64_t chunk_size = numel_per_tensor;

    // Allocate output tensor: shape [num_chunks, ...original shape]
    std::vector<int64_t> out_shape;
    out_shape.push_back(num_chunks);
    out_shape.insert(out_shape.end(), shape.begin(), shape.end());
    auto output = torch::empty(out_shape, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    TORCH_CHECK(output.is_contiguous(), "Output tensor should be contiguous by default");

    // Collect device pointers
    std::vector<const float*> host_ptrs;
    host_ptrs.reserve(num_chunks);
    for (const auto& t : cuda_tensors) {
        host_ptrs.push_back(t.data_ptr<float>());
    }

    // Allocate device memory for pointer array
    const float** device_ptrs = nullptr;
    cudaError_t err = cudaMalloc(&device_ptrs, num_chunks * sizeof(float*));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed for pointer array");

    // Copy pointer array to device
    err = cudaMemcpy(device_ptrs, host_ptrs.data(), num_chunks * sizeof(float*), cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy of pointer array failed");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch stack kernel
    stack_forward_launcher(device_ptrs, output.data_ptr<float>(), chunk_size, num_chunks, stream);

    // Free device pointer array after kernel completes (asynchronous, but safe to free immediately since kernel only reads it; however, to be safe, we can synchronize)
    cudaStreamSynchronize(stream);
    cudaFree(device_ptrs);

    return output;
}

// bce_with_logits_loss_forward: input and target must be CUDA float32 tensors of same shape.
torch::Tensor bce_with_logits_loss_forward(const torch::Tensor& input, const torch::Tensor& target) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have the same shape");
    int64_t n = input.numel();
    TORCH_CHECK(n > 0, "input and target cannot be empty");

    // Allocate scalar output
    auto output = torch::empty({}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch BCE loss launcher
    bce_with_logits_loss_forward_launcher(input.data_ptr<float>(), target.data_ptr<float>(), output.data_ptr<float>(), n, stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("stack_forward", &stack_forward, py::arg("tensors"));
    m.def("bce_with_logits_loss_forward", &bce_with_logits_loss_forward, py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
