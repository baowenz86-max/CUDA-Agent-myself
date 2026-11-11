// Binding for L1 loss mean reduction CUDA kernel.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void l1_loss_forward_launcher(
    const float* input,
    const float* target,
    float* output,
    int64_t input_numel,
    int64_t ndim,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* target_shape,
    const int64_t* target_strides,
    cudaStream_t stream);

// PyTorch wrapper: computes mean(|input - target|)
torch::Tensor l1_loss_forward(torch::Tensor input, torch::Tensor target) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");

    // Handle broadcasting: make target same shape as input via expand
    auto target_expanded = target.expand(input.sizes());

    // Ensure contiguous tensors for direct pointer access
    auto input_contig = input.contiguous();
    auto target_contig = target_expanded.contiguous();

    // Output scalar tensor on CUDA
    auto output = torch::empty({}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract shape and strides (after expansion and contiguous)
    int64_t ndim = input_contig.dim();
    std::vector<int64_t> input_shape(ndim);
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> target_shape(ndim);
    std::vector<int64_t> target_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        input_shape[i] = input_contig.size(i);
        input_strides[i] = input_contig.stride(i);
        target_shape[i] = target_contig.size(i);
        target_strides[i] = target_contig.stride(i);
    }

    // Call launcher
    l1_loss_forward_launcher(
        input_contig.data_ptr<float>(),
        target_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        input_contig.numel(),
        ndim,
        input_shape.data(),
        input_strides.data(),
        target_shape.data(),
        target_strides.data(),
        stream
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_l1_loss_forward(pybind11::module& m) {
    m.def("l1_loss_forward", &l1_loss_forward,
          "Compute mean absolute error (L1 loss) between two CUDA tensors",
          pybind11::arg("input"),
          pybind11::arg("target"));
}

REGISTER_BINDING(l1_loss_forward, register_l1_loss_forward);
