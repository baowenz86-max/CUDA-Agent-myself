#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_pad_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream);

extern "C" void custom_cat_dim1_launcher(
    const float* a,
    const float* b,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t a_N_stride, int64_t a_C_stride, int64_t a_H_stride, int64_t a_W_stride,
    int64_t b_N_stride, int64_t b_C_stride, int64_t b_H_stride, int64_t b_W_stride,
    cudaStream_t stream);

// custom_pad wrapper
torch::Tensor custom_pad_forward(torch::Tensor x, int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "custom_pad: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_pad: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_pad: input must be 4D (N, C, H, W)");
    
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;
    
    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    int64_t in_N_stride = x.stride(0);
    int64_t in_C_stride = x.stride(1);
    int64_t in_H_stride = x.stride(2);
    int64_t in_W_stride = x.stride(3);
    
    custom_pad_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride,
        stream
    );
    
    return output;
}

// custom_cat wrapper (dim=1 specialization)
torch::Tensor custom_cat_forward(torch::Tensor a, torch::Tensor b, int64_t dim) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "custom_cat: inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "custom_cat: inputs must be float32");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "custom_cat: inputs must be 4D tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "custom_cat: input shapes must match");
    TORCH_CHECK(dim >= 0 && dim < 4, "custom_cat: dim must be in [0, 3]");
    // This task only requires dim=1
    TORCH_CHECK(dim == 1, "custom_cat: only dim=1 is supported in this implementation");
    
    int64_t N = a.size(0);
    int64_t C = a.size(1);
    int64_t H = a.size(2);
    int64_t W = a.size(3);
    
    auto output = torch::empty({N, 2 * C, H, W}, a.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    int64_t a_N_stride = a.stride(0);
    int64_t a_C_stride = a.stride(1);
    int64_t a_H_stride = a.stride(2);
    int64_t a_W_stride = a.stride(3);
    int64_t b_N_stride = b.stride(0);
    int64_t b_C_stride = b.stride(1);
    int64_t b_H_stride = b.stride(2);
    int64_t b_W_stride = b.stride(3);
    
    custom_cat_dim1_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        a_N_stride, a_C_stride, a_H_stride, a_W_stride,
        b_N_stride, b_C_stride, b_H_stride, b_W_stride,
        stream
    );
    
    return output;
}

// registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_pad", &custom_pad_forward, "ConstantPad2d replacement",
          py::arg("x"), py::arg("pad_left"), py::arg("pad_right"), py::arg("pad_top"), py::arg("pad_bottom"));
    m.def("custom_cat", &custom_cat_forward, "Concatenate along dim",
          py::arg("a"), py::arg("b"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
