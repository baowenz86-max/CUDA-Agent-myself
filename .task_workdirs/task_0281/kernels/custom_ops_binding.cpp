#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

#include <vector>
#include <cmath>

// Declare launchers defined in custom_ops.cu
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int numel,
    const int* shape,
    const int* in_strides,
    const int* out_strides,
    int ndim,
    cudaStream_t stream);

extern "C" void avg_pool1d_forward_launcher(
    float* output,
    const float* input,
    int N,
    int C,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    const int* in_strides,
    const int* out_strides,
    cudaStream_t stream);

// Wrapper for relu_forward
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    TORCH_CHECK(x.dim() >= 1 && x.dim() <= 3, "relu_forward: input must be 1D, 2D, or 3D");

    // Allocate output with same sizes and strides as input to preserve stride semantics.
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int ndim = x.dim();
    int numel = x.numel();

    // Create small host arrays for shape, in_strides, out_strides; launcher will copy to device internally
    // Actually launcher expects device pointers? The .cu launcher takes const int* pointers, which could be host or device.
    // We'll allocate pinned/host arrays and pass them; but kernel launches need device data. Since the .cu launcher does not copy,
    // it expects device pointers. We can pass host pointers if the kernel uses them directly? The kernel code accesses shape[], in_strides[], out_strides[] as if they are in global memory accessible by device, so they must be device pointers.
    // We'll allocate small device tensors to hold these arrays, copy from host, and pass device pointers to launcher.
    // This avoids modifying .cu.

    std::vector<int> shape(ndim);
    std::vector<int> in_strides(ndim);
    std::vector<int> out_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
    }

    // Allocate device memory for these arrays using torch::empty with int32 options on same device.
    auto shape_t = torch::empty({ndim}, x.options().dtype(torch::kInt32));
    auto in_strides_t = torch::empty({ndim}, x.options().dtype(torch::kInt32));
    auto out_strides_t = torch::empty({ndim}, x.options().dtype(torch::kInt32));

    // Copy host vectors to device tensors using memcpy via cudaMemcpy (since we can't use torch:: operations for compute).
    cudaMemcpy(shape_t.data_ptr<int>(), shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_strides_t.data_ptr<int>(), in_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_strides_t.data_ptr<int>(), out_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        shape_t.data_ptr<int>(),
        in_strides_t.data_ptr<int>(),
        out_strides_t.data_ptr<int>(),
        ndim,
        stream);

    return output;
}

// Wrapper for avg_pool1d_forward
torch::Tensor avg_pool1d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "avg_pool1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avg_pool1d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "avg_pool1d_forward: input must be 3D (N, C, L)");
    TORCH_CHECK(kernel_size > 0 && stride > 0, "kernel_size and stride must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int L_in = x.size(2);

    // Compute output length
    int L_out = static_cast<int>(std::floor((L_in + 2 * padding - kernel_size) / static_cast<double>(stride))) + 1;

    // Allocate contiguous output tensor [N, C, L_out]
    auto output = torch::empty({N, C, L_out}, x.options());

    // Prepare strides arrays (input may have arbitrary strides, output is contiguous)
    std::vector<int> in_strides(3);
    std::vector<int> out_strides(3);
    for (int i = 0; i < 3; ++i) {
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
    }

    // Allocate device memory for strides
    auto in_strides_t = torch::empty({3}, x.options().dtype(torch::kInt32));
    auto out_strides_t = torch::empty({3}, x.options().dtype(torch::kInt32));

    cudaMemcpy(in_strides_t.data_ptr<int>(), in_strides.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_strides_t.data_ptr<int>(), out_strides.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool1d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        C,
        L_in,
        L_out,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        in_strides_t.data_ptr<int>(),
        out_strides_t.data_ptr<int>(),
        stream);

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward", py::arg("x"));
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "AvgPool1d forward",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
