// Binding for custom ops declared in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Extern C launcher declarations matching custom_ops.cu
extern "C" void one_hot_float_launcher(const int64_t* input_cpu,
                                        int64_t N,
                                        int64_t H,
                                        int64_t W,
                                        int num_classes,
                                        float* output_cuda,
                                        cudaStream_t stream);

extern "C" void one_hot_int64_launcher(const int64_t* input_cpu,
                                        int64_t N,
                                        int64_t H,
                                        int64_t W,
                                        int num_classes,
                                        int64_t* output_cuda,
                                        cudaStream_t stream);

extern "C" void cast_float32_launcher(const int64_t* input_cuda,
                                       float* output_cuda,
                                       int64_t total_elements,
                                       cudaStream_t stream);

extern "C" void pad_constant_launcher(const float* input_cuda,
                                      float* output_cuda,
                                      int64_t total_output,
                                      int N, int H_in, int W_in, int C,
                                      int pad_left, int pad_right,
                                      int pad_top, int pad_bottom,
                                      float value,
                                      cudaStream_t stream);

extern "C" void softmax_dim_launcher(const float* input_cuda,
                                     float* output_cuda,
                                     int64_t rows,
                                     int C,
                                     cudaStream_t stream);

// Wrapper for one_hot_float
torch::Tensor one_hot_float(torch::Tensor x, int64_t num_classes) {
    TORCH_CHECK(x.is_cpu(), "Input x must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "Input x must be int64");
    TORCH_CHECK(x.dim() == 3, "Input x must be 3D [batch, height, width]");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    int64_t N = x.size(0);
    int64_t H = x.size(1);
    int64_t W = x.size(2);

    // Allocate output on CUDA: [N, H, W, num_classes]
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA);
    auto output = torch::empty({N, H, W, num_classes}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    one_hot_float_launcher(
        x.data_ptr<int64_t>(),
        N, H, W,
        static_cast<int>(num_classes),
        output.data_ptr<float>(),
        stream);

    return output;
}

// Wrapper for one_hot_int64
torch::Tensor one_hot_int64(torch::Tensor x, int64_t num_classes) {
    TORCH_CHECK(x.is_cpu(), "Input x must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "Input x must be int64");
    TORCH_CHECK(x.dim() == 3, "Input x must be 3D [batch, height, width]");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    int64_t N = x.size(0);
    int64_t H = x.size(1);
    int64_t W = x.size(2);

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt64)
                       .device(torch::kCUDA);
    auto output = torch::empty({N, H, W, num_classes}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    one_hot_int64_launcher(
        x.data_ptr<int64_t>(),
        N, H, W,
        static_cast<int>(num_classes),
        output.data_ptr<int64_t>(),
        stream);

    return output;
}

// Wrapper for cast_float32
torch::Tensor cast_float32(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "Input x must be int64");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");

    int64_t total = x.numel();
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA);
    auto output = torch::empty_like(x, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cast_float32_launcher(
        x.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

// Wrapper for pad_constant
torch::Tensor pad_constant(torch::Tensor x,
                           int64_t pad_left,
                           int64_t pad_right,
                           int64_t pad_top,
                           int64_t pad_bottom,
                           double value) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.dim() == 4, "Input x must be 4D [N, H, W, C]");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding sizes must be non-negative");

    int64_t N = x.size(0);
    int64_t H_in = x.size(1);
    int64_t W_in = x.size(2);
    int64_t C = x.size(3);

    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    auto output = torch::empty({N, H_out, W_out, C}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pad_constant_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N * H_out * W_out * C,
        static_cast<int>(N),
        static_cast<int>(H_in),
        static_cast<int>(W_in),
        static_cast<int>(C),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        static_cast<float>(value),
        stream);

    return output;
}

// Wrapper for softmax_dim
torch::Tensor softmax_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim must be a valid dimension");

    // For this task, dim is 1 which corresponds to last dimension after permutation.
    // The custom kernel handles softmax along the last dimension (C).
    // If dim is not the last dimension, we would need a different kernel,
    // but according to contract it is always last dim in this task.
    TORCH_CHECK(dim == x.dim() - 1, "Only softmax over last dimension is supported by this binding");

    int64_t rows = x.numel() / x.size(-1);
    int64_t C = x.size(-1);
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        static_cast<int>(C),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("one_hot_float", &one_hot_float,
          "One-hot encode int64 CPU tensor to float32 CUDA tensor",
          py::arg("x"), py::arg("num_classes"));
    m.def("pad_constant", &pad_constant,
          "Constant pad height and width dimensions",
          py::arg("x"), py::arg("pad_left"), py::arg("pad_right"),
          py::arg("pad_top"), py::arg("pad_bottom"), py::arg("value"));
    m.def("softmax_dim", &softmax_dim,
          "Softmax along specified dimension",
          py::arg("x"), py::arg("dim"));
    m.def("one_hot_int64", &one_hot_int64,
          "One-hot encode int64 CPU tensor to int64 CUDA tensor (debug)",
          py::arg("x"), py::arg("num_classes"));
    m.def("cast_float32", &cast_float32,
          "Cast int64 tensor to float32 (debug)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
