#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_sigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    const long long* input_strides,
    const long long* output_strides,
    const long long* sizes,
    int ndim,
    cudaStream_t stream
);

extern "C" void custom_multiply_scalar_launcher(
    float* output,
    const float* input,
    long long numel,
    float scalar,
    const long long* input_strides,
    const long long* output_strides,
    const long long* sizes,
    int ndim,
    cudaStream_t stream
);

namespace {

// Helper: allocate device memory for strides/sizes, copy from host, and free after call
class DeviceStridesSizes {
public:
    DeviceStridesSizes(const torch::Tensor& tensor) {
        ndim = tensor.dim();
        sizes_host.resize(ndim);
        in_strides_host.resize(ndim);
        out_strides_host.resize(ndim);

        // Get sizes and strides
        for (int d = 0; d < ndim; ++d) {
            sizes_host[d] = tensor.size(d);
            in_strides_host[d] = tensor.stride(d);
            out_strides_host[d] = tensor.stride(d); // output same shape, same strides
        }

        // Allocate device arrays
        cudaMalloc((void**)&sizes_dev, ndim * sizeof(long long));
        cudaMalloc((void**)&in_strides_dev, ndim * sizeof(long long));
        cudaMalloc((void**)&out_strides_dev, ndim * sizeof(long long));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaMemcpyAsync(sizes_dev, sizes_host.data(), ndim * sizeof(long long), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(in_strides_dev, in_strides_host.data(), ndim * sizeof(long long), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_strides_dev, out_strides_host.data(), ndim * sizeof(long long), cudaMemcpyHostToDevice, stream);
    }

    ~DeviceStridesSizes() {
        cudaFree(sizes_dev);
        cudaFree(in_strides_dev);
        cudaFree(out_strides_dev);
    }

    long long* sizes_ptr() { return sizes_dev; }
    long long* in_strides_ptr() { return in_strides_dev; }
    long long* out_strides_ptr() { return out_strides_dev; }
    int dim() const { return ndim; }

private:
    int ndim;
    std::vector<long long> sizes_host;
    std::vector<long long> in_strides_host;
    std::vector<long long> out_strides_host;
    long long* sizes_dev = nullptr;
    long long* in_strides_dev = nullptr;
    long long* out_strides_dev = nullptr;
};

} // anonymous namespace

// Wrapper for custom_sigmoid
torch::Tensor custom_sigmoid(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    DeviceStridesSizes strides_sizes(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_sigmoid_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        strides_sizes.in_strides_ptr(),
        strides_sizes.out_strides_ptr(),
        strides_sizes.sizes_ptr(),
        strides_sizes.dim(),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sigmoid kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for custom_multiply_scalar
torch::Tensor custom_multiply_scalar(torch::Tensor x, double scalar = 2.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    DeviceStridesSizes strides_sizes(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_multiply_scalar_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        static_cast<float>(scalar),
        strides_sizes.in_strides_ptr(),
        strides_sizes.out_strides_ptr(),
        strides_sizes.sizes_ptr(),
        strides_sizes.dim(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_multiply_scalar kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_sigmoid", &custom_sigmoid, "Sigmoid activation (CUDA)",
          py::arg("x"));
    m.def("custom_multiply_scalar", &custom_multiply_scalar, "Multiply tensor by scalar (CUDA)",
          py::arg("x"), py::arg("scalar") = 2.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
