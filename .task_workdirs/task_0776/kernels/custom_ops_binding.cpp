// Binding for adaptive_avg_pool1d and repeat_interleave_like
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int b,
    int c,
    int l_in,
    int l_out,
    cudaStream_t stream
);

extern "C" void repeat_interleave_like_launcher(
    const float* input,
    float* output,
    long long input_size,
    int repeats,
    cudaStream_t stream
);

// Wrapper for adaptive_avg_pool1d
torch::Tensor adaptive_avg_pool1d(torch::Tensor input, int64_t output_size) {
    // Parameter checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    // Ensure contiguity
    input = input.contiguous();

    int b = input.size(0);
    int c = input.size(1);
    int l_in = input.size(2);
    int l_out = static_cast<int>(output_size);

    auto output = torch::empty({b, c, l_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool1d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        b,
        c,
        l_in,
        l_out,
        stream
    );

    return output;
}

// Wrapper for repeat_interleave_like
torch::Tensor repeat_interleave_like(torch::Tensor input, int64_t repeats, int64_t dim) {
    // Parameter checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(repeats > 0, "repeats must be positive");

    // Normalize dim
    int ndim = input.dim();
    if (dim < 0) {
        dim = dim + ndim + 1;
    }
    // The launcher only supports stack-like copy in contiguous layout for dim=1 on 3D input,
    // which is exactly the model usage. We enforce that here to avoid silent incorrectness.
    TORCH_CHECK(ndim == 3, "repeat_interleave_like currently supports only 3D input");
    TORCH_CHECK(dim == 1, "repeat_interleave_like currently supports only dim=1");

    // Ensure contiguity
    input = input.contiguous();

    long long input_size = static_cast<long long>(input.numel());
    auto output = torch::empty({input.size(0), repeats, input.size(1), input.size(2)}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    repeat_interleave_like_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        static_cast<int>(repeats),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool1d", &adaptive_avg_pool1d,
          "Adaptive average pooling 1D",
          py::arg("input"), py::arg("output_size"));

    m.def("repeat_interleave_like", &repeat_interleave_like,
          "Repeat interleave like torch.stack",
          py::arg("input"), py::arg("repeats"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
