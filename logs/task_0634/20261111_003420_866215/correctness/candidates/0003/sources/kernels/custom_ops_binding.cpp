#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

extern "C" void to_cuda_launcher(const float* cpu_input, float* gpu_output, int64_t numel, cudaStream_t stream);
extern "C" void mean_reduce_keepdim_launcher(const float* x, float* mean, int batch_size, int channels, int height, int width, cudaStream_t stream);
extern "C" void where_less_zero_launcher(const float* x, const float* mean, float* output, int batch_size, int channels, int height, int width, cudaStream_t stream);

torch::Tensor to_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu(), "to_cuda: input must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "to_cuda: input must be float32");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options().device(torch::kCUDA));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    to_cuda_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);
    return output;
}

torch::Tensor mean_reduce_keepdim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "mean_reduce_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_reduce_keepdim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_reduce_keepdim: input must be 4D");

    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto output = torch::empty({batch_size, 1, 1, 1}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_reduce_keepdim_launcher(x.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, stream);
    return output;
}

torch::Tensor where_less_zero(torch::Tensor x, torch::Tensor mean) {
    TORCH_CHECK(x.is_cuda() && mean.is_cuda(), "where_less_zero: both inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && mean.dtype() == torch::kFloat32, "where_less_zero: inputs must be float32");
    TORCH_CHECK(x.dim() == 4, "where_less_zero: x must be 4D");
    TORCH_CHECK(mean.dim() == 4 && mean.size(0) == x.size(0) && mean.size(1) == 1 && mean.size(2) == 1 && mean.size(3) == 1,
                "where_less_zero: mean must be of shape [batch,1,1,1]");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());
    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    where_less_zero_launcher(x.data_ptr<float>(), mean.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("to_cuda", &to_cuda, "Copy CPU tensor to current CUDA device", py::arg("x"));
    m.def("mean_reduce_keepdim", &mean_reduce_keepdim, "Compute mean over dims [1,2,3] with keepdim", py::arg("x"), py::arg("dim"));
    m.def("where_less_zero", &where_less_zero, "Apply threshold: x < mean ? 0 : x", py::arg("x"), py::arg("mean"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
