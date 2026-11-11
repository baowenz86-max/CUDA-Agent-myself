#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(float* out, const float* in, int64_t total, cudaStream_t stream);
extern "C" void argmax_onehot_forward_launcher(
    const float* x,
    int64_t* out,
    int64_t batch,
    int64_t classes,
    int64_t height,
    int64_t width,
    int64_t num_classes,
    cudaStream_t stream);

// abs_forward: element-wise absolute value
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "abs_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "abs_forward: input must be float32");

    // Ensure contiguous; launcher assumes linear access
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t total = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        total,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// argmax_onehot_forward: argmax over dim=1 then one-hot encode
torch::Tensor argmax_onehot_forward(torch::Tensor x, int64_t num_classes) {
    TORCH_CHECK(x.is_cuda(), "argmax_onehot_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "argmax_onehot_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "argmax_onehot_forward: input must be 4D [batch, classes, height, width]");

    int64_t batch = x.size(0);
    int64_t classes = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    TORCH_CHECK(classes > 0, "argmax_onehot_forward: classes dimension must be positive");
    TORCH_CHECK(num_classes >= classes, "argmax_onehot_forward: num_classes must be >= classes");

    // Ensure contiguous input
    auto x_contig = x.contiguous();

    // Allocate output shape [batch, height, width, num_classes]
    auto output = torch::empty({batch, height, width, num_classes},
                               x_contig.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    argmax_onehot_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        batch,
        classes,
        height,
        width,
        num_classes,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "argmax_onehot_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value",
          pybind11::arg("x"));
    m.def("argmax_onehot_forward", &argmax_onehot_forward,
          "Argmax over dim=1 and one-hot encode",
          pybind11::arg("x"), pybind11::arg("num_classes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
