#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* input,
                                     float* output,
                                     long long numel,
                                     int dim0, int dim1, int dim2,
                                     long long stride0, long long stride1, long long stride2,
                                     cudaStream_t stream);

extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      long long numel,
                                      int dim0, int dim1, int dim2,
                                      long long stride0, long long stride1, long long stride2,
                                      cudaStream_t stream);

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "gelu_forward: input must be 3D");

    // Ensure contiguous for simplicity while the launcher supports arbitrary strides
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int dim0 = x_contig.size(0);
    int dim1 = x_contig.size(1);
    int dim2 = x_contig.size(2);
    long long stride0 = x_contig.stride(0);
    long long stride1 = x_contig.stride(1);
    long long stride2 = x_contig.stride(2);
    long long numel = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        dim0, dim1, dim2,
        stride0, stride1, stride2,
        stream
    );

    return output;
}

torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "tanh_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "tanh_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "tanh_forward: input must be 3D");

    // Ensure contiguous for simplicity while the launcher supports arbitrary strides
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int dim0 = x_contig.size(0);
    int dim1 = x_contig.size(1);
    int dim2 = x_contig.size(2);
    long long stride0 = x_contig.stride(0);
    long long stride1 = x_contig.stride(1);
    long long stride2 = x_contig.stride(2);
    long long numel = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tanh_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        dim0, dim1, dim2,
        stride0, stride1, stride2,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "Exact GELU activation (erf-based)",
          pybind11::arg("x"));
    m.def("tanh_forward", &tanh_forward, "Hyperbolic tangent activation",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
