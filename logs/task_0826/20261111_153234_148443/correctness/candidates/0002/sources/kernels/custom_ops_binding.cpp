#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clone_to_cuda_launcher(const float* input, float* output, int64_t total, cudaStream_t stream);
extern "C" void rot90_1_2_launcher(const float* input, float* output,
                                   int64_t B, int64_t C, int64_t H, int64_t W, cudaStream_t stream);

static torch::Tensor clone_to_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clone_to_cuda expects float32 tensor");
    TORCH_CHECK(x.is_contiguous(), "clone_to_cuda expects contiguous input");

    auto output = torch::empty(x.sizes(), x.options().device(torch::kCUDA));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (x.is_cuda()) {
        // Already on CUDA, perform device-to-device clone
        cudaMemcpyAsync(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        // CPU to CUDA
        cudaMemcpyAsync(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.numel() * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "clone_to_cuda copy failed");
    return output;
}

static torch::Tensor rot90_1_2(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "rot90_1_2 expects a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rot90_1_2 expects float32 tensor");
    TORCH_CHECK(x.dim() == 4, "rot90_1_2 expects 4D tensor");
    TORCH_CHECK(x.is_contiguous(), "rot90_1_2 expects contiguous input");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output shape [B, W, C, H]
    auto output = torch::empty({B, W, C, H}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rot90_1_2_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                       B, C, H, W, stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rot90_1_2 kernel launch failed");

    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("clone_to_cuda", &clone_to_cuda, "Clone tensor to CUDA", py::arg("x"));
    m.def("rot90_1_2", &rot90_1_2, "Rotate tensor 90 degrees in dims (1,2)", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);