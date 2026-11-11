#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration matching custom_ops.cu
extern "C" void replication_pad2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_height,
    int64_t stride_width,
    cudaStream_t stream);

torch::Tensor replication_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    // Validate tensor properties
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (NCHW)");

    // Optional contiguous check not required because launcher supports strides

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    // Allocate contiguous output tensor on same device
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with strides from input (which may be non-contiguous)
    replication_pad2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_left,
        pad_top,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream);

    return output;
}

void register_replication_pad2d(pybind11::module& m) {
    m.def("replication_pad2d_forward", &replication_pad2d_forward,
          "ReplicationPad2d forward (CUDA)",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
}

REGISTER_BINDING(replication_pad2d, register_replication_pad2d);
