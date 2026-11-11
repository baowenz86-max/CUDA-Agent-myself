#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reduce_sum_launcher(const float* input, float* output, double* partial_sums,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream);

extern "C" void round_launcher(const float* input, float* output, int numel, cudaStream_t stream);

// Wrapper for torch.sum over dims (2,3) for a 4D contiguous float32 tensor.
torch::Tensor custom_reduce_sum(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim0 == 2 && dim1 == 3, "Only reduction over last two dims is supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({N, C}, x.options());

    // Allocate temporary buffer for partial sums (double) on device
    int num_blocks_per_output = 16;
    auto partial_sums = torch::empty({N * C, num_blocks_per_output}, x.options().dtype(torch::kFloat64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reduce_sum_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_sums.data_ptr<double>(),
        N, C, H, W,
        stream
    );

    return output;
}

// Wrapper for torch.round
torch::Tensor custom_round(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    round_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_reduce_sum", &custom_reduce_sum,
          "Reduce sum over last two dimensions",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"));
    m.def("custom_round", &custom_round,
          "Round half to even",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
