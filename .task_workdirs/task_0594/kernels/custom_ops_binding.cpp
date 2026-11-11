#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t batch,
    int64_t C1,
    int64_t C2,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void custom_equal_launcher(
    const float* x1,
    const float* x2,
    int* result,
    int64_t numel,
    cudaStream_t stream);

// custom_cat: concatenate two tensors along dimension 1 (assumes 4D tensors)
static torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "custom_cat: x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "custom_cat: x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "custom_cat: x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "custom_cat: x2 must be float32");
    TORCH_CHECK(dim == 1, "custom_cat: only dim=1 is supported for 4D tensors");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "custom_cat: expected 4D tensors");
    TORCH_CHECK(x1.size(0) == x2.size(0), "custom_cat: batch size mismatch");
    TORCH_CHECK(x1.size(2) == x2.size(2), "custom_cat: H dimension mismatch");
    TORCH_CHECK(x1.size(3) == x2.size(3), "custom_cat: W dimension mismatch");

    // Ensure inputs are contiguous for the launcher's flat memory access
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    int64_t batch = x1_contig.size(0);
    int64_t C1 = x1_contig.size(1);
    int64_t C2 = x2_contig.size(1);
    int64_t H = x1_contig.size(2);
    int64_t W = x1_contig.size(3);

    // Allocate output with shape [batch, C1+C2, H, W]
    auto out = torch::empty({batch, C1 + C2, H, W}, x1_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_cat_launcher(
        out.data_ptr<float>(),
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        batch,
        C1,
        C2,
        H,
        W,
        stream
    );

    return out;
}

// custom_equal: compare two tensors element-wise, return Python bool
static bool custom_equal(torch::Tensor x1, torch::Tensor x2) {
    TORCH_CHECK(x1.is_cuda(), "custom_equal: x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "custom_equal: x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "custom_equal: x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "custom_equal: x2 must be float32");
    TORCH_CHECK(x1.numel() == x2.numel(), "custom_equal: tensors must have the same number of elements");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "custom_equal: tensors must have the same shape");

    // Ensure contiguous for flat memory access
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    int64_t numel = x1_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Device int for result, initialized to 1 (meaning equal)
    int* result_device;
    cudaMalloc(&result_device, sizeof(int));
    cudaMemsetAsync(result_device, 1, sizeof(int), stream);

    custom_equal_launcher(
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        result_device,
        numel,
        stream
    );

    // Copy result back to host
    int result_host = 0;
    cudaMemcpyAsync(&result_host, result_device, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(result_device);

    return result_host != 0;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat", &custom_cat,
          "Concatenate two float32 CUDA tensors along dimension 1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("custom_equal", &custom_equal,
          "Compare two float32 CUDA tensors for exact equality",
          py::arg("x1"), py::arg("x2"));
}

// Auto-register binding
REGISTER_BINDING(custom_ops, register_custom_ops);
