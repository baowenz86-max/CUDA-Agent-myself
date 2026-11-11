#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void replication_pad2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

torch::Tensor replication_pad2d_forward(
    torch::Tensor x,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding must be non-negative");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int H_out = H_in + pad_top + pad_bottom;
    const int W_out = W_in + pad_left + pad_right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    replication_pad2d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H_in, W_in,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        stream);

    return output;
}

void register_replication_pad2d(pybind11::module& m) {
    m.def("replication_pad2d_forward", &replication_pad2d_forward,
          "ReplicationPad2d forward",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));
}

REGISTER_BINDING(replication_pad2d, register_replication_pad2d);
