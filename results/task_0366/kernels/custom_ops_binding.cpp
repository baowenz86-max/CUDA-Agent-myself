#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" bool custom_equal_cpu_int64_launcher(
    const int64_t* x,
    const int64_t* y,
    int64_t numel,
    cudaStream_t stream);

extern "C" bool custom_sin_cuda_int64_launcher(
    const int64_t* cpu_input,
    int64_t numel,
    float* cuda_output,
    cudaStream_t stream);

bool custom_equal_cpu_int64(torch::Tensor x, torch::Tensor y) {
    // Accept any int64 tensors, regardless of device.
    TORCH_CHECK(x.dtype() == torch::kInt64, "x must be int64");
    TORCH_CHECK(y.dtype() == torch::kInt64, "y must be int64");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    int64_t numel = x.numel();
    if (numel == 0) {
        return true;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (x.is_cuda() || y.is_cuda()) {
        // If either tensor is on CUDA, move both to CPU for comparison.
        auto x_cpu = x.to(torch::kCPU).contiguous();
        auto y_cpu = y.to(torch::kCPU).contiguous();
        return custom_equal_cpu_int64_launcher(
            x_cpu.data_ptr<int64_t>(),
            y_cpu.data_ptr<int64_t>(),
            numel,
            stream);
    } else {
        // Both on CPU, ensure contiguous.
        auto x_contig = x.contiguous();
        auto y_contig = y.contiguous();
        return custom_equal_cpu_int64_launcher(
            x_contig.data_ptr<int64_t>(),
            y_contig.data_ptr<int64_t>(),
            numel,
            stream);
    }
}

torch::Tensor custom_sin_cuda_int64(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kInt64, "x must be int64");

    int64_t numel = x.numel();
    auto output = torch::empty_like(x, x.options().dtype(torch::kFloat32).device(torch::kCUDA));
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (x.is_cuda()) {
        // Input is already on CUDA, use directly.
        int64_t* d_input = x.data_ptr<int64_t>();
        bool success = custom_sin_cuda_int64_launcher(
            d_input,
            numel,
            output.data_ptr<float>(),
            stream);
        TORCH_CHECK(success, "custom_sin_cuda_int64 launcher failed");
    } else {
        // Input on CPU, copy to CUDA then compute.
        auto x_cuda = x.to(torch::kCUDA).contiguous();
        bool success = custom_sin_cuda_int64_launcher(
            x_cuda.data_ptr<int64_t>(),
            numel,
            output.data_ptr<float>(),
            stream);
        TORCH_CHECK(success, "custom_sin_cuda_int64 launcher failed");
    }

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_equal_cpu_int64", &custom_equal_cpu_int64,
          "Compare two int64 tensors element-wise (torch.equal)",
          py::arg("x"), py::arg("y"));
    m.def("custom_sin_cuda_int64", &custom_sin_cuda_int64,
          "Compute sin of int64 tensor, producing float32 CUDA tensor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
