#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream);

torch::Tensor mean_dim0(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C, H, W]");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape: [C, H, W]
    auto output = torch::empty({C, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim0_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

void register_mean_dim0(pybind11::module& m) {
    m.def("mean_dim0", &mean_dim0, "Compute mean over dim=0 for a 4D tensor",
          py::arg("x"));
}

REGISTER_BINDING(mean_dim0, register_mean_dim0);
