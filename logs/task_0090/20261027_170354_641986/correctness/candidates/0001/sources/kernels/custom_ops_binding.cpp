#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 中导出的 launcher
extern "C" void elu_forward_launcher(const float* input, float* output, long long n, cudaStream_t stream);
extern "C" void sigmoid_forward_launcher(const float* input, float* output, long long n, cudaStream_t stream);
extern "C" void elu_sigmoid_forward_launcher(const float* input, float* output, long long n, cudaStream_t stream);

static torch::Tensor check_and_contiguous(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
    if (!x.is_contiguous()) {
        return x.contiguous();
    }
    return x;
}

torch::Tensor elu_forward(torch::Tensor x) {
    auto x_contig = check_and_contiguous(x, "x");
    auto y = torch::empty_like(x_contig);
    long long n = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(x_contig.data_ptr<float>(), y.data_ptr<float>(), n, stream);
    return y;
}

torch::Tensor sigmoid_forward(torch::Tensor x) {
    auto x_contig = check_and_contiguous(x, "x");
    auto y = torch::empty_like(x_contig);
    long long n = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_forward_launcher(x_contig.data_ptr<float>(), y.data_ptr<float>(), n, stream);
    return y;
}

torch::Tensor elu_sigmoid_forward(torch::Tensor x) {
    auto x_contig = check_and_contiguous(x, "x");
    auto y = torch::empty_like(x_contig);
    long long n = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_sigmoid_forward_launcher(x_contig.data_ptr<float>(), y.data_ptr<float>(), n, stream);
    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU forward");
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward");
    m.def("elu_sigmoid_forward", &elu_sigmoid_forward, "Fused ELU + Sigmoid forward");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
