#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

extern "C" void cosine_similarity_forward_launcher(
    const float* x1,
    const float* x2,
    float* output,
    int B,
    int C,
    int HW,
    long long stride_b1,
    long long stride_c1,
    long long stride_b2,
    long long stride_c2,
    long long out_numel,
    cudaStream_t stream);

torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sigmoid_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "sigmoid_forward: input must be float32");

    // Make contiguous if needed; launcher assumes contiguous data
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x_contig.numel()),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor cosine_similarity_forward(torch::Tensor x1, torch::Tensor x2) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "cosine_similarity_forward: inputs must be CUDA tensors");
    TORCH_CHECK(x1.scalar_type() == torch::kFloat32 && x2.scalar_type() == torch::kFloat32,
                "cosine_similarity_forward: inputs must be float32");
    TORCH_CHECK(x1.dim() >= 3 && x2.dim() >= 3, "cosine_similarity_forward: inputs must have at least 3 dimensions");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "cosine_similarity_forward: input shapes must match");

    // Make contiguous
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    int B = static_cast<int>(x1_contig.size(0));
    int C = static_cast<int>(x1_contig.size(1));
    int HW = 1;
    for (int i = 2; i < x1_contig.dim(); ++i) {
        HW *= static_cast<int>(x1_contig.size(i));
    }

    // Output shape [B, HW] (all dims from 2 onward flattened)
    std::vector<int64_t> out_sizes;
    out_sizes.push_back(B);
    for (int i = 2; i < x1_contig.dim(); ++i) {
        out_sizes.push_back(x1_contig.size(i));
    }
    auto output = torch::empty(out_sizes, x1_contig.options());

    long long stride_b = static_cast<long long>(C) * HW;
    long long stride_c = HW;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cosine_similarity_forward_launcher(
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, HW,
        stride_b, stride_c,
        stride_b, stride_c,
        static_cast<long long>(B) * HW,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cosine_similarity_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward (CUDA)",
          pybind11::arg("x"));
    m.def("cosine_similarity_forward", &cosine_similarity_forward,
          "Cosine similarity forward (CUDA)",
          pybind11::arg("x1"), pybind11::arg("x2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
