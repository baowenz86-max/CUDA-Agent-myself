#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void copy_to_cuda_launcher(float* d_out,
                                      const float* h_in,
                                      size_t numel,
                                      cudaStream_t stream);

extern "C" void max_step_launcher(float* d_values,
                                  int64_t* d_indices,
                                  const float* d_input,
                                  int N, int H, int W,
                                  cudaStream_t stream);

extern "C" void view_as_complex_launcher(cuComplex* d_out,
                                         const float* d_in,
                                         int N, int H, int W,
                                         cudaStream_t stream);

// Helper to get current CUDA stream
cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// copy_to_cuda: CPU -> CUDA copy
torch::Tensor copy_to_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cpu(), "copy_to_cuda expects a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "copy_to_cuda expects float32 tensor");
    TORCH_CHECK(x.is_contiguous(), "copy_to_cuda expects contiguous tensor");

    auto out = torch::empty_like(x, x.options().device(torch::kCUDA));
    size_t numel = x.numel();
    cudaStream_t stream = get_current_stream();

    copy_to_cuda_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    return out;
}

// max_step: torch.max(x, dim=1, keepdim=True)
torch::return_types::max max_step(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "max_step expects a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_step expects float32 tensor");
    TORCH_CHECK(x.is_contiguous(), "max_step expects contiguous tensor");
    TORCH_CHECK(x.dim() == 5, "max_step expects 5D tensor");
    TORCH_CHECK(x.size(1) == 2, "max_step expects dim 1 size to be 2");
    TORCH_CHECK(x.size(4) == 2, "max_step expects last dim size to be 2");

    int64_t N = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto values = torch::empty({N, 1, H, W, 2}, x.options());
    auto indices = torch::empty({N, 1, H, W, 2}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = get_current_stream();
    max_step_launcher(
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    return torch::return_types::max(values, indices);
}

// squeeze_step: x.squeeze(1), returns a tensor shape [N,H,W,2]
torch::Tensor squeeze_step(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "squeeze_step expects a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "squeeze_step expects 5D tensor");
    TORCH_CHECK(x.size(1) == 1, "squeeze_step expects dim 1 size to be 1");

    // Return a view; if not contiguous later, view_as_complex_step will handle
    return x.squeeze(1);
}

// view_as_complex_step: torch.view_as_complex(x)
torch::Tensor view_as_complex_step(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "view_as_complex_step expects a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "view_as_complex_step expects float32 tensor");
    TORCH_CHECK(x.dim() == 4, "view_as_complex_step expects 4D tensor");
    TORCH_CHECK(x.size(3) == 2, "view_as_complex_step expects last dim size 2");

    // Ensure contiguous for the kernel's linear indexing
    auto x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t H = x_contig.size(1);
    int64_t W = x_contig.size(2);

    auto out = torch::empty({N, H, W}, x_contig.options().dtype(torch::kComplexFloat));

    cudaStream_t stream = get_current_stream();
    view_as_complex_launcher(
        reinterpret_cast<cuComplex*>(out.data_ptr<c10::complex<float>>()),
        x_contig.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("copy_to_cuda", &copy_to_cuda, "Copy CPU tensor to CUDA",
          py::arg("x"));
    m.def("max_step", &max_step, "Max along dim 1, keepdim=True",
          py::arg("x"));
    m.def("squeeze_step", &squeeze_step, "Squeeze dim 1",
          py::arg("x"));
    m.def("view_as_complex_step", &view_as_complex_step, "View as complex",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
