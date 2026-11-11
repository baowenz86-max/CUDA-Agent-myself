#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

extern "C" void max_pool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_size, int64_t stride,
    cudaStream_t stream);

extern "C" void stack_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

torch::Tensor max_pool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride) {
    TORCH_CHECK(input.is_cuda(), "max_pool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "max_pool2d_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "max_pool2d_forward: input must be 4D [N, C, H, W]");
    TORCH_CHECK(kernel_size > 0, "max_pool2d_forward: kernel_size must be positive");
    TORCH_CHECK(stride > 0, "max_pool2d_forward: stride must be positive");

    // Ensure contiguous
    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t H_out = (H - kernel_size) / stride + 1;
    int64_t W_out = (W - kernel_size) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "max_pool2d_forward: output size must be positive");

    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_pool2d_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor stack_forward(std::vector<torch::Tensor> tensors) {
    TORCH_CHECK(!tensors.empty(), "stack_forward: tensors list must not be empty");

    const auto& first = tensors[0];
    TORCH_CHECK(first.is_cuda(), "stack_forward: all tensors must be CUDA");
    TORCH_CHECK(first.dtype() == torch::kFloat32, "stack_forward: all tensors must be float32");

    // Ensure all tensors have same shape, dtype, device
    for (const auto& t : tensors) {
        TORCH_CHECK(t.is_cuda(), "stack_forward: all tensors must be CUDA");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "stack_forward: all tensors must be float32");
        TORCH_CHECK(t.sizes() == first.sizes(), "stack_forward: all tensors must have same shape");
    }

    int64_t num_tensors = static_cast<int64_t>(tensors.size());
    std::vector<int64_t> output_shape;
    output_shape.reserve(1 + first.dim());
    output_shape.push_back(num_tensors);
    for (int64_t i = 0; i < first.dim(); ++i) {
        output_shape.push_back(first.size(i));
    }

    auto output = torch::empty(output_shape, first.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Each input tensor is copied to the corresponding slice of the output.
    // We assume each tensor is contiguous (or we make it contiguous).
    for (int64_t i = 0; i < num_tensors; ++i) {
        auto t_contig = tensors[i].contiguous();
        int64_t numel = t_contig.numel();
        if (numel == 0) continue;
        float* out_ptr = output.data_ptr<float>() + i * numel;
        stack_forward_launcher(
            t_contig.data_ptr<float>(),
            out_ptr,
            numel,
            stream);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "stack_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"));
    m.def("stack_forward", &stack_forward,
          "Stack tensors along new dimension 0",
          pybind11::arg("tensors"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
