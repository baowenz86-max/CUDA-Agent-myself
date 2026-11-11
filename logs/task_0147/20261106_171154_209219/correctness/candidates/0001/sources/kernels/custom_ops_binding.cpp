#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void elu_forward_launcher(
    float* output,
    const float* input,
    int rank,
    const int64_t* shape,
    const int64_t* input_strides,
    const int64_t* output_strides,
    int64_t numel,
    cudaStream_t stream
);

static torch::Tensor elu_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 1, "Input must have at least one dimension");

    // Ensure contiguous (kernel handles arbitrary strides, but for simplicity and performance
    // we make it contiguous; however, launcher supports strides so we could also pass non-contig
    // but here we follow common pattern of contiguous input)
    auto input = x.contiguous();
    auto output = torch::empty_like(input);

    int64_t numel = input.numel();
    int rank = input.dim();

    // Prepare shape and strides arrays on CPU (small tensors)
    auto shape_cpu = input.sizes().vec();  // std::vector<int64_t>
    auto input_strides_cpu = input.strides().vec();
    auto output_strides_cpu = output.strides().vec();

    // Allocate device memory for shape and strides
    auto shape_tensor = torch::from_blob(shape_cpu.data(), {rank}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).to(torch::kCUDA);
    auto input_strides_tensor = torch::from_blob(input_strides_cpu.data(), {rank}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).to(torch::kCUDA);
    auto output_strides_tensor = torch::from_blob(output_strides_cpu.data(), {rank}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).to(torch::kCUDA);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    elu_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        rank,
        shape_tensor.data_ptr<int64_t>(),
        input_strides_tensor.data_ptr<int64_t>(),
        output_strides_tensor.data_ptr<int64_t>(),
        numel,
        stream
    );

    return output;
}

void register_elu_forward(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU forward (alpha=1.0)",
          pybind11::arg("x"));
}

REGISTER_BINDING(elu_forward, register_elu_forward);
