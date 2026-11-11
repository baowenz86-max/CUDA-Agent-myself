#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void greater_than_launcher(
    const float* x1,
    const float* x2,
    bool* out,
    long long N, long long C, long long H, long long W,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3,
    cudaStream_t stream);

extern "C" void concatenate_dim1_launcher(
    const float* x1,
    const float* x2,
    float* out,
    long long N, long long C1, long long C2, long long H, long long W,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3,
    cudaStream_t stream);

// Wrapper for element-wise greater-than
torch::Tensor greater_than(torch::Tensor x1, torch::Tensor x2) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4, "x1 must be 4D");
    TORCH_CHECK(x2.dim() == 4, "x2 must be 4D");
    TORCH_CHECK(x1.size(0) == x2.size(0) && x1.size(1) == x2.size(1) &&
                x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3),
                "input shapes must match");

    long long N = x1.size(0);
    long long C = x1.size(1);
    long long H = x1.size(2);
    long long W = x1.size(3);

    auto out = torch::empty({N, C, H, W}, x1.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract strides as long long (int64)
    long long sx1[4] = {x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3)};
    long long sx2[4] = {x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3)};
    long long sout[4] = {out.stride(0), out.stride(1), out.stride(2), out.stride(3)};

    greater_than_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<bool>(),
        N, C, H, W,
        sx1[0], sx1[1], sx1[2], sx1[3],
        sx2[0], sx2[1], sx2[2], sx2[3],
        sout[0], sout[1], sout[2], sout[3],
        stream
    );

    return out;
}

// Wrapper for concatenation along dimension 1
torch::Tensor concatenate_dim1(torch::Tensor x1, torch::Tensor x2) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4, "x1 must be 4D");
    TORCH_CHECK(x2.dim() == 4, "x2 must be 4D");
    TORCH_CHECK(x1.size(0) == x2.size(0) && x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3),
                "batch, height, width must match");

    long long N = x1.size(0);
    long long C1 = x1.size(1);
    long long C2 = x2.size(1);
    long long H = x1.size(2);
    long long W = x1.size(3);

    auto out = torch::empty({N, C1 + C2, H, W}, x1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long sx1[4] = {x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3)};
    long long sx2[4] = {x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3)};
    long long sout[4] = {out.stride(0), out.stride(1), out.stride(2), out.stride(3)};

    concatenate_dim1_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C1, C2, H, W,
        sx1[0], sx1[1], sx1[2], sx1[3],
        sx2[0], sx2[1], sx2[2], sx2[3],
        sout[0], sout[1], sout[2], sout[3],
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("greater_than", &greater_than, "Element-wise greater-than comparison",
          py::arg("x1"), py::arg("x2"));
    m.def("concatenate_dim1", &concatenate_dim1, "Concatenate two tensors along dimension 1",
          py::arg("x1"), py::arg("x2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
