#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void custom_max_view_as_complex_launcher(
    const float* x,
    float2* complex_out,
    long long* indices,
    int batch,
    int height,
    int width,
    int channels,
    cudaStream_t stream);

static std::tuple<torch::Tensor, torch::Tensor> custom_max_view_as_complex(
    torch::Tensor x)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "custom_max_view_as_complex: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_view_as_complex: input must be float32");
    TORCH_CHECK(x.dim() == 5, "custom_max_view_as_complex: input must have 5 dimensions [batch, channels, height, width, 2]");

    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int last = x.size(4);

    TORCH_CHECK(channels == 2, "custom_max_view_as_complex: channel dimension must be 2");
    TORCH_CHECK(last == 2, "custom_max_view_as_complex: last dimension must be 2");
    TORCH_CHECK(batch > 0 && height > 0 && width > 0, "custom_max_view_as_complex: batch, height, width must be positive");

    // Ensure contiguous input (the kernel assumes contiguous layout)
    auto x_contig = x.contiguous();

    // Allocate outputs
    auto complex_out = torch::empty({batch, height, width},
                                    torch::dtype(torch::kComplexFloat).device(x.device()));
    auto indices = torch::empty({batch, 1, height, width, 2},
                                torch::dtype(torch::kLong).device(x.device()));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_max_view_as_complex_launcher(
        x_contig.data_ptr<float>(),
        reinterpret_cast<float2*>(complex_out.data_ptr<c10::complex<float>>()),
        indices.data_ptr<int64_t>(),
        batch,
        height,
        width,
        channels,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_view_as_complex: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(complex_out, indices);
}

static void register_custom_max_view_as_complex(pybind11::module& m) {
    m.def("custom_max_view_as_complex", &custom_max_view_as_complex,
          "Fused max over dim=1 and view_as_complex",
          py::arg("x"));
}

REGISTER_BINDING(custom_max_view_as_complex, register_custom_max_view_as_complex);
