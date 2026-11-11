// PyTorch bindings for custom ops defined in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Launcher declarations from custom_ops.cu
extern "C" void softmax_dim1_forward_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void hardtanh_forward_launcher(
    const float* x,
    float* out,
    long long total_elems,
    int ndim,
    const int64_t* dims,
    const int64_t* istride,
    const int64_t* ostride,
    cudaStream_t stream);

// Softmax along dim=1 for 4D tensor
torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "softmax_dim1_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax_dim1_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "softmax_dim1_forward: input must be 4D [N, C, H, W]");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Pass strides as long long (elements)
    long long stride_n = x.stride(0);
    long long stride_c = x.stride(1);
    long long stride_h = x.stride(2);
    long long stride_w = x.stride(3);

    long long out_stride_n = out.stride(0);
    long long out_stride_c = out.stride(1);
    long long out_stride_h = out.stride(2);
    long long out_stride_w = out.stride(3);

    softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return out;
}

// Hardtanh element-wise
torch::Tensor hardtanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardtanh_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardtanh_forward: input must be float32");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long total_elems = x.numel();
    int ndim = x.dim();

    std::vector<int64_t> dims_vec;
    std::vector<int64_t> istride_vec;
    std::vector<int64_t> ostride_vec;
    for (int i = 0; i < ndim; ++i) {
        dims_vec.push_back(x.size(i));
        istride_vec.push_back(x.stride(i));
        ostride_vec.push_back(out.stride(i));
    }

    // Allocate device tensors for dims, strides
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto dims_cuda = torch::empty({ndim}, options);
    auto istride_cuda = torch::empty({ndim}, options);
    auto ostride_cuda = torch::empty({ndim}, options);

    // Create CPU tensors from vectors and copy to CUDA tensors
    auto cpu_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto dims_cpu = torch::from_blob(dims_vec.data(), {ndim}, cpu_options);
    auto istride_cpu = torch::from_blob(istride_vec.data(), {ndim}, cpu_options);
    auto ostride_cpu = torch::from_blob(ostride_vec.data(), {ndim}, cpu_options);

    dims_cuda.copy_(dims_cpu);
    istride_cuda.copy_(istride_cpu);
    ostride_cuda.copy_(ostride_cpu);

    hardtanh_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elems,
        ndim,
        dims_cuda.data_ptr<int64_t>(),
        istride_cuda.data_ptr<int64_t>(),
        ostride_cuda.data_ptr<int64_t>(),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax along dim=1 for 4D float32 tensor", py::arg("x"));
    m.def("hardtanh_forward", &hardtanh_forward, "Hardtanh element-wise clamp to [-1, 1]", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
