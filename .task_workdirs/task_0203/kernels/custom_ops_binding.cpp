#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../binding_registry.h"

extern "C" void gelu_forward_launcher(const float* input,
                                       float* output,
                                       const long long* shape,
                                       const long long* input_strides,
                                       int rank,
                                       long long total_elements,
                                       cudaStream_t stream);

extern "C" void trunc_forward_launcher(const float* input,
                                       float* output,
                                       const long long* shape,
                                       const long long* input_strides,
                                       int rank,
                                       long long total_elements,
                                       cudaStream_t stream);

namespace {

torch::Tensor validate_and_prepare(const torch::Tensor& x, const std::string& op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.numel() > 0, op_name, ": input must be non-empty");
    return x.contiguous();
}

std::vector<long long> get_shape(const torch::Tensor& t) {
    std::vector<long long> shape(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        shape[i] = t.size(i);
    }
    return shape;
}

std::vector<long long> get_strides(const torch::Tensor& t) {
    std::vector<long long> strides(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        strides[i] = t.stride(i);
    }
    return strides;
}

torch::Tensor launch_forward(
    const torch::Tensor& x_original,
    const std::string& op_name,
    void (*launcher)(const float*, float*, const long long*, const long long*, int, long long, cudaStream_t)) {
    torch::Tensor x = validate_and_prepare(x_original, op_name);
    auto shape_vec = get_shape(x);
    auto strides_vec = get_strides(x);
    int rank = x.dim();
    long long total = x.numel();

    auto output = torch::empty_like(x);

    // Copy shape and strides to device
    long long* d_shape;
    long long* d_strides;
    cudaMalloc(&d_shape, rank * sizeof(long long));
    cudaMalloc(&d_strides, rank * sizeof(long long));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(d_shape, shape_vec.data(), rank * sizeof(long long), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, strides_vec.data(), rank * sizeof(long long), cudaMemcpyHostToDevice, stream);

    launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        d_shape,
        d_strides,
        rank,
        total,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, ": CUDA error: ", cudaGetErrorString(err));

    cudaFree(d_shape);
    cudaFree(d_strides);

    return output;
}

torch::Tensor gelu_forward(torch::Tensor x) {
    return launch_forward(x, "gelu_forward", gelu_forward_launcher);
}

torch::Tensor trunc_forward(torch::Tensor x) {
    return launch_forward(x, "trunc_forward", trunc_forward_launcher);
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward", py::arg("x"));
    m.def("trunc_forward", &trunc_forward, "Truncation forward", py::arg("x"));
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
