// Binding for channel_shuffle_cuda
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void channel_shuffle_cuda_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W, int groups,
    cudaStream_t stream);

namespace {

inline void check_input(const torch::Tensor& x, int64_t groups, int64_t C) {
    TORCH_CHECK(x.is_cuda(), "channel_shuffle_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "channel_shuffle_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "channel_shuffle_cuda: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "channel_shuffle_cuda: input must be 4D (N,C,H,W)");
    TORCH_CHECK(C % groups == 0, "channel_shuffle_cuda: number of channels must be divisible by groups");
    TORCH_CHECK(groups > 0, "channel_shuffle_cuda: groups must be positive");
}

} // anonymous namespace

torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    check_input(x, groups, C);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    channel_shuffle_cuda_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(groups),
        stream
    );

    return output;
}

void register_channel_shuffle_cuda(pybind11::module& m) {
    m.def("channel_shuffle_cuda", &channel_shuffle_cuda,
          "Channel shuffle operation on CUDA",
          pybind11::arg("x"), pybind11::arg("groups"));
}

REGISTER_BINDING(channel_shuffle_cuda, register_channel_shuffle_cuda);
