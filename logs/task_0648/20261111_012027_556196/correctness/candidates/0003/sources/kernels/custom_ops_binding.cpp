#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_reduce_op_launcher(const float* input,
                                        float* output,
                                        int64_t batch,
                                        int64_t inner_size,
                                        int64_t batch_stride,
                                        cudaStream_t stream);

extern "C" void remainder_op_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      double divisor,
                                      cudaStream_t stream);

// mean_reduce_op: reduce over dims (1,2,3) keeping dims, equivalent to torch.mean(x, dim=(1,2,3), keepdim=True).
// The kernel only supports this specific reduction for 4D float32 tensors.
torch::Tensor mean_reduce_op(torch::Tensor x, std::vector<int64_t> dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "mean_reduce_op: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_reduce_op: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_reduce_op: input must be 4-dimensional");
    TORCH_CHECK(dim == std::vector<int64_t>({1, 2, 3}), "mean_reduce_op: only dim=(1,2,3) reduction is supported");
    TORCH_CHECK(keepdim, "mean_reduce_op: keepdim must be true");

    int64_t batch = x.size(0);
    int64_t inner_size = x.size(1) * x.size(2) * x.size(3);
    int64_t batch_stride = x.stride(0);

    auto output = torch::empty({batch, 1, 1, 1}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_reduce_op_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        inner_size,
        batch_stride,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_reduce_op: kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// remainder_op: element-wise remainder with float divisor, equivalent to torch.remainder(x, divisor).
torch::Tensor remainder_op(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "remainder_op: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "remainder_op: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "remainder_op: input must be contiguous");

    int64_t numel = x.numel();
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    remainder_op_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        divisor,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "remainder_op: kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("mean_reduce_op", &mean_reduce_op,
          "Mean reduce over channel and spatial dimensions",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("remainder_op", &remainder_op,
          "Element-wise remainder",
          py::arg("x"), py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
