#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void mul_forward_launcher(
    float* out,
    const float* x,
    const float* y,
    int ndim,
    const int64_t* shape,
    const int64_t* stride_out,
    const int64_t* stride_x,
    const int64_t* stride_y,
    int64_t numel,
    cudaStream_t stream);

torch::Tensor mul_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    // Ensure tensors are contiguous for simpler stride handling; contract allows supporting strides,
    // but our launcher handles arbitrary strides, so we can use the actual strides.
    // We do not require contiguity, but for simplicity we can use .contiguous() to make strides standard.
    // However, to preserve exact reference behavior for non-contiguous inputs, we should pass actual strides.
    // The input tensors are expected to be contiguous from generation, but we will handle general case.
    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();

    auto output = torch::empty_like(x_contig);

    int64_t numel = x_contig.numel();
    int ndim = x_contig.dim();

    // Prepare shape and strides as int64_t arrays on host, then copy to device for launcher.
    // The launcher expects device pointers for these arrays (defined as const int64_t* __restrict__).
    // We allocate device memory using cudaMalloc.
    std::vector<int64_t> shape_host(ndim);
    std::vector<int64_t> stride_out_host(ndim);
    std::vector<int64_t> stride_x_host(ndim);
    std::vector<int64_t> stride_y_host(ndim);

    for (int i = 0; i < ndim; ++i) {
        shape_host[i] = x_contig.size(i);
        stride_out_host[i] = output.stride(i);
        stride_x_host[i] = x_contig.stride(i);
        stride_y_host[i] = y_contig.stride(i);
    }

    int64_t* d_shape;
    int64_t* d_stride_out;
    int64_t* d_stride_x;
    int64_t* d_stride_y;

    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride_out, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride_x, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride_y, ndim * sizeof(int64_t));

    cudaMemcpyAsync(d_shape, shape_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream().stream());
    cudaMemcpyAsync(d_stride_out, stride_out_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream().stream());
    cudaMemcpyAsync(d_stride_x, stride_x_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream().stream());
    cudaMemcpyAsync(d_stride_y, stride_y_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream().stream());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mul_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        ndim,
        d_shape,
        d_stride_out,
        d_stride_x,
        d_stride_y,
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mul_forward kernel launch failed: ", cudaGetErrorString(err));

    cudaFree(d_shape);
    cudaFree(d_stride_out);
    cudaFree(d_stride_x);
    cudaFree(d_stride_y);

    return output;
}

void register_mul_forward(pybind11::module& m) {
    m.def("mul_forward", &mul_forward, "Elementwise multiplication of two CUDA tensors",
          py::arg("x"), py::arg("y"));
}

REGISTER_BINDING(mul_forward, register_mul_forward);
