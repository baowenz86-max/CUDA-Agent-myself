#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void rot90_clone_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

torch::Tensor rot90_clone_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [batch, channels, height, width]");

    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    // Allocate base tensor [batch, channels, width, height] contiguous.
    // We will write the rotated/cloned data into this base, then return a permuted view
    // with shape [batch, width, channels, height] and strides [307200, 320, 102400, 1].
    auto base = torch::empty({batch, channels, width, height}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    rot90_clone_launcher(
        x.data_ptr<float>(),
        base.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    // Return view with expected shape and strides: permute dims (0,2,1,3)
    return base.permute({0, 2, 1, 3});
}

void register_rot90_clone(pybind11::module& m) {
    m.def("rot90_clone_forward", &rot90_clone_forward,
          "Clone and rotate 90 degrees (k=1, dims=(1,2))",
          py::arg("x"));
}

REGISTER_BINDING(rot90_clone, register_rot90_clone);
