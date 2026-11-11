#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_channel_fmod_launcher(const float* input,
                                           float* output,
                                           int batch,
                                           int channels,
                                           int height,
                                           int width,
                                           float divisor,
                                           cudaStream_t stream);

extern "C" void fmod_scalar_launcher(const float* input,
                                     float* output,
                                     size_t N,
                                     float modulus,
                                     cudaStream_t stream);

// Helper to validate float32 CUDA tensor
static void check_float_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.numel() > 0, name, " must have non-zero number of elements");
}

torch::Tensor mean_channel_fmod(torch::Tensor x, double divisor) {
    check_float_cuda(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (batch, channels, height, width)");

    // Ensure contiguous for launcher
    auto x_contig = x.contiguous();

    int batch = x_contig.size(0);
    int channels = x_contig.size(1);
    int height = x_contig.size(2);
    int width = x_contig.size(3);

    // Allocate output [batch, 1, height, width]
    auto output = torch::empty({batch, 1, height, width}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_channel_fmod_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        static_cast<float>(divisor),
        stream
    );

    return output;
}

torch::Tensor fmod_scalar(torch::Tensor x, double modulus) {
    check_float_cuda(x, "x");
    // Work on contiguous input
    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fmod_scalar_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<size_t>(x_contig.numel()),
        static_cast<float>(modulus),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("mean_channel_fmod", &mean_channel_fmod,
          "Mean over channel dimension (keepdim=True) followed by fmod with divisor",
          py::arg("x"), py::arg("divisor"));
    m.def("fmod_scalar", &fmod_scalar,
          "Element-wise fmod with scalar modulus",
          py::arg("x"), py::arg("modulus"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);