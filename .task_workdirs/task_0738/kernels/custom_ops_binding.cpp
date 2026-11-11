#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_stack_launcher(const float* x,
                                     const float* y,
                                     float* out,
                                     long long n_per_input,
                                     cudaStream_t stream);

extern "C" void custom_clone_launcher(const float* in,
                                      float* out,
                                      long long numel,
                                      cudaStream_t stream);

// Helper to ensure CUDA float32 contiguous tensor
torch::Tensor check_and_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    return t.contiguous();
}

torch::Tensor custom_stack(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "custom_stack inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
                "custom_stack inputs must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "custom_stack inputs must have identical shape");

    // Ensure contiguous copies; inputs in reference are contiguous.
    x = x.contiguous();
    y = y.contiguous();

    auto sizes = x.sizes();
    int64_t inner_numel = x.numel();

    // New tensor shape: [2, *input_sizes]
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(sizes.size() + 1);
    out_sizes.push_back(2);
    for (auto s : sizes) {
        out_sizes.push_back(s);
    }

    auto out = torch::empty(out_sizes, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_stack_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        (long long)inner_numel,
        stream
    );

    return out;
}

torch::Tensor custom_clone(torch::Tensor x) {
    auto x_contig = check_and_contiguous(x, "custom_clone input");

    auto out = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_clone_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        (long long)x_contig.numel(),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_stack", &custom_stack,
          "Stack two tensors along a new leading dimension",
          py::arg("x"), py::arg("y"));
    m.def("custom_clone", &custom_clone,
          "Clone a contiguous tensor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
