#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_logical_or_launcher(const float* x,
                                            const float* y,
                                            bool* out,
                                            long long numel,
                                            cudaStream_t stream);

extern "C" void custom_cast_float_launcher(const bool* in,
                                            float* out,
                                            long long numel,
                                            cudaStream_t stream);

extern "C" void custom_mse_loss_mean_launcher(const float* input,
                                               const float* target,
                                               float* output,
                                               long long numel,
                                               cudaStream_t stream);

namespace {

cudaStream_t get_cuda_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

void check_cuda_error(const char* name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " CUDA error: ", cudaGetErrorString(err));
}

} // namespace

// -----------------------------------------------------------------------------
// custom_logical_or binding
// -----------------------------------------------------------------------------
torch::Tensor custom_logical_or(torch::Tensor x, torch::Tensor y, double threshold) {
    TORCH_CHECK(x.is_cuda(), "custom_logical_or: x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "custom_logical_or: y must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_logical_or: x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "custom_logical_or: y must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_logical_or: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "custom_logical_or: y must be contiguous");
    TORCH_CHECK(x.numel() == y.numel(), "custom_logical_or: x and y must have same numel");
    TORCH_CHECK(x.numel() > 0, "custom_logical_or: input must not be empty");

    auto out = torch::empty_like(x, x.options().dtype(torch::kBool));

    cudaStream_t stream = get_cuda_stream();
    custom_logical_or_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<bool>(),
        static_cast<long long>(x.numel()),
        stream
    );
    check_cuda_error("custom_logical_or_launcher");

    return out;
}

// -----------------------------------------------------------------------------
// custom_cast_float binding
// -----------------------------------------------------------------------------
torch::Tensor custom_cast_float(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_cast_float: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBool, "custom_cast_float: x must be bool");
    TORCH_CHECK(x.is_contiguous(), "custom_cast_float: x must be contiguous");
    TORCH_CHECK(x.numel() > 0, "custom_cast_float: input must not be empty");

    auto out = torch::empty_like(x, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = get_cuda_stream();
    custom_cast_float_launcher(
        x.data_ptr<bool>(),
        out.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );
    check_cuda_error("custom_cast_float_launcher");

    return out;
}

// -----------------------------------------------------------------------------
// custom_mse_loss_mean binding
// -----------------------------------------------------------------------------
torch::Tensor custom_mse_loss_mean(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "custom_mse_loss_mean: input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "custom_mse_loss_mean: target must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "custom_mse_loss_mean: input must be float32");
    TORCH_CHECK(target.scalar_type() == torch::kFloat32, "custom_mse_loss_mean: target must be float32");
    TORCH_CHECK(input.is_contiguous(), "custom_mse_loss_mean: input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "custom_mse_loss_mean: target must be contiguous");
    TORCH_CHECK(input.numel() == target.numel(), "custom_mse_loss_mean: input and target must have same numel");
    TORCH_CHECK(input.numel() > 0, "custom_mse_loss_mean: input must not be empty");

    // 输出为标量张量，形状为 []
    auto output = torch::empty({}, input.options().dtype(torch::kFloat32));

    cudaStream_t stream = get_cuda_stream();
    custom_mse_loss_mean_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(input.numel()),
        stream
    );
    check_cuda_error("custom_mse_loss_mean_launcher");

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_logical_or", &custom_logical_or,
          "Elementwise logical OR of (x > threshold) and (y > threshold)",
          py::arg("x"), py::arg("y"), py::arg("threshold") = 0.5);
    m.def("custom_cast_float", &custom_cast_float,
          "Cast bool tensor to float32",
          py::arg("x"));
    m.def("custom_mse_loss_mean", &custom_mse_loss_mean,
          "Mean squared error loss",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
