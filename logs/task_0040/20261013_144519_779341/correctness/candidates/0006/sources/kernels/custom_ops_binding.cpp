#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_bias_broadcast_launcher(
    float* output,
    const float* input,
    const float* bias,
    int batch,
    int out_channels,
    int Lout,
    cudaStream_t stream);

extern "C" void roll_dim2_launcher(
    float* output,
    const float* input,
    int batch,
    int out_channels,
    int Lout,
    int shift,
    cudaStream_t stream);

extern "C" void view_flatten_launcher(
    float* output,
    const float* input,
    int total,
    cudaStream_t stream);

// Helper to check CUDA errors after launch
void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// add_bias_broadcast
torch::Tensor add_bias_broadcast(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, out_channels, Lout]");
    TORCH_CHECK(bias.dim() == 2, "Bias must be 2D [out_channels, 1]");
    TORCH_CHECK(bias.size(0) == input.size(1), "Bias channels must match input channels");
    TORCH_CHECK(bias.size(1) == 1, "Bias second dimension must be 1");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");

    auto output = torch::empty_like(input);

    int batch = static_cast<int>(input.size(0));
    int out_channels = static_cast<int>(input.size(1));
    int Lout = static_cast<int>(input.size(2));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_broadcast_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch,
        out_channels,
        Lout,
        stream
    );
    check_cuda_error("add_bias_broadcast kernel");
    return output;
}

// roll_dim2
torch::Tensor roll_dim2(torch::Tensor input, int64_t shift) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, out_channels, Lout]");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(shift > 0, "shift must be positive for this implementation");

    auto output = torch::empty_like(input);

    int batch = static_cast<int>(input.size(0));
    int out_channels = static_cast<int>(input.size(1));
    int Lout = static_cast<int>(input.size(2));
    int shift_int = static_cast<int>(shift);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    roll_dim2_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        out_channels,
        Lout,
        shift_int,
        stream
    );
    check_cuda_error("roll_dim2 kernel");
    return output;
}

// view_flatten
torch::Tensor view_flatten(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, out_channels, Lout]");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t batch = input.size(0);
    int64_t out_channels = input.size(1);
    int64_t Lout = input.size(2);
    int64_t num_features = out_channels * Lout;
    auto output = torch::empty({batch, num_features}, input.options());

    int total = static_cast<int>(input.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    view_flatten_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        total,
        stream
    );
    check_cuda_error("view_flatten kernel");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("add_bias_broadcast", &add_bias_broadcast, "Add bias broadcast");
    m.def("roll_dim2", &roll_dim2, "Roll along dim 2", py::arg("input"), py::arg("shift"));
    m.def("view_flatten", &view_flatten, "Flatten view");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
