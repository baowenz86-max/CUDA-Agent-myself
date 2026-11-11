#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void custom_max_dim1_launcher(
    float* values,
    int64_t* indices,
    const float* input,
    int B, int C, int H, int W,
    cudaStream_t stream);

extern "C" void custom_stack_dim1_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    cudaStream_t stream);

torch::Tensor custom_max_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    auto x_contig = x.contiguous();
    int B = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    auto values = torch::empty({B, H, W}, x_contig.options());
    auto indices = torch::empty({B, H, W}, x_contig.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_max_dim1_launcher(
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        x_contig.data_ptr<float>(),
        B, C, H, W,
        stream);

    return values;
}

torch::Tensor custom_stack_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    auto x_contig = x.contiguous();
    int B = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    auto output = torch::empty({B, 2, C, H, W}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_stack_dim1_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        B, C, H, W,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1, "Computes max values over dim=1",
          py::arg("x"));
    m.def("custom_stack_dim1", &custom_stack_dim1, "Computes torch.stack((x,x), dim=1)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
