#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int C,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream);

extern "C" void custom_relu_launcher(
    const float* x,
    float* output,
    int batch,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Implementation of custom_max_dim1
// Returns torch.return_types.max with values and indices
py::object custom_max_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (batch, C, H, W)");

    int batch = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensors: values float32, indices int64, shape [batch, 1, H, W], contiguous
    auto values = torch::empty({batch, 1, H, W}, x.options());
    auto indices = torch::empty({batch, 1, H, W}, x.options().dtype(torch::kInt64));

    // Get strides (input may be non-contiguous)
    long long x_stride_b = x.stride(0);
    long long x_stride_c = x.stride(1);
    long long x_stride_h = x.stride(2);
    long long x_stride_w = x.stride(3);
    long long out_stride_b = values.stride(0);
    long long out_stride_h = values.stride(2);  // stride for dimension 1 is not used in kernel
    long long out_stride_w = values.stride(3);

    cudaStream_t stream = getCurrentStream();

    custom_max_dim1_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch, C, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_h, out_stride_w,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_dim1 kernel launch failed: ", cudaGetErrorString(err));

    // Construct torch.return_types.max object using a dict to satisfy Python API
    py::dict result_dict;
    result_dict["values"] = values;
    result_dict["indices"] = indices;
    py::object max_type = py::module::import("torch").attr("return_types").attr("max");
    return max_type(result_dict);
}

// Implementation of custom_relu
// Returns a tensor
torch::Tensor custom_relu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (batch, 1, H, W)");

    int batch = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor with same shape as input, contiguous
    auto output = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    long long x_stride_b = x.stride(0);
    long long x_stride_c = x.stride(1);
    long long x_stride_h = x.stride(2);
    long long x_stride_w = x.stride(3);
    long long out_stride_b = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = getCurrentStream();

    custom_relu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_relu kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1, "Custom max over dim=1 (keepdim=True) returning max values and indices",
          py::arg("x"));
    m.def("custom_relu", &custom_relu, "Custom ReLU activation",
          py::arg("x"));
}

// Auto-register binding
REGISTER_BINDING(custom_ops, register_custom_ops);
