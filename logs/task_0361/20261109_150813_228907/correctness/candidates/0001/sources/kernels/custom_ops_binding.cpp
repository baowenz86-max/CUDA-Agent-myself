#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void hardswish_forward_launcher(
    float* output,
    const float* input,
    size_t N,
    size_t in_stride,
    size_t out_stride,
    cudaStream_t stream
);

// PyTorch wrapper for hardswish
torch::Tensor hardswish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto output = torch::empty_like(x);

    // Compute strides in element units.
    // For contiguous tensors, these are 1.
    // For non-contiguous (e.g., transposed), use the actual data pointer differences
    // scaled by sizeof(float). We assume the storage is valid and the tensor may have
    // arbitrary strides; the kernel uses a linear index over N elements and accesses
    // in[i * in_stride] and out[i * out_stride]. This matches strides provided that
    // the underlying memory layout is consistent.
    size_t N = static_cast<size_t>(x.numel());
    size_t in_stride = 1;
    size_t out_stride = 1;

    if (!x.is_contiguous()) {
        // For non-contiguous tensors, compute the element stride as the distance
        // between consecutive elements in the innermost dimension.
        // We assume the innermost stride is the last element of x.strides() and
        // that the kernel's linear index maps through that stride.
        // However, a general non-contiguous tensor may not be representable by a single
        // stride. For safety and simplicity, we can make the tensor contiguous if it isn't.
        // But the contract says "layout: supports strides"; we should preserve strides.
        // Let's compute in_stride = x.strides()[x.dim()-1] (innermost stride in elements)
        // and out_stride = output.strides()[x.dim()-1].
        // But the kernel assumes a 1D linear layout per innermost stride; for higher-dim
        // tensors with non-contiguous memory, a single stride is insufficient.
        // Given that the reference input is contiguous and the launcher accepts strides,
        // we can support the common case of contiguous or simple strided tensors by
        // taking the innermost stride. If more complex, we could fall back to contiguous,
        // but that would change strides and violate the contract.
        // Here we follow the launcher's intended use: for non-contiguous, the caller
        // should ensure the tensor is contiguous or has a consistent innermost stride.
        // To be robust, we call .contiguous() on input if it's not already contiguous,
        // and then output will be contiguous.
        // But that would copy and potentially change strides; the contract says preserve strides,
        // so we should not force contiguous. However, the launcher is designed with strides,
        // and we can pass the innermost strides. For the common 2D/3D case with stride pattern
        // [h*w, w, 1], the innermost stride is 1, so in_stride=1 and out_stride=1; the kernel
        // will then access linearly, which is correct for contiguous tensors. For non-contiguous
        // but still regular stride patterns, we can compute the effective element stride as
        // x.strides()[x.dim()-1] (or the minimal stride), but to handle all cases we need a more
        // general indexing. Since the reference input is contiguous, and the contract only
        // requires supporting strides, we can leave the stride computation simple: if tensor
        // is not contiguous, we'll make it contiguous for safety, but that changes output strides.
        // Let's check the launcher: it multiplies linear index i by in_stride to get element offset.
        // This implies that the memory layout is a single contiguous block with stride in_stride,
        // i.e., the tensor is effectively 1D with stride in_stride. That only works if the tensor
        // is contiguous (stride 1) or if it's a slice with constant stride (e.g., from narrow).
        // In this task, the input is generated as torch.randn(16, 512, 64000) which is contiguous.
        // So we can assume contiguous for the reference. To be safe, we check if x.is_contiguous(),
        // and if not, we call x = x.contiguous(). This will produce correct output but will not
        // preserve strides. However, the contract only requires shape and values to match, not
        // exact strides? Let's check: returns has "shape": "same as x". No mention of strides.
        // In the task_contract.json, output stride is specified as [32768000, 64000, 1], which
        // corresponds to contiguous. So output should be contiguous. So we can simply ensure
        // both input and output are contiguous. That simplifies.
        // Therefore, we will make x contiguous if it is not, and output will be contiguous
        // because empty_like of contiguous input gives contiguous output.
        // We then set in_stride = 1 and out_stride = 1.
        // This is acceptable because the reference input is contiguous.
        x = x.contiguous();
        in_stride = 1;
    }

    // If x was contiguous originally, empty_like output is also contiguous, so strides = 1.
    // If x was made contiguous, output is contiguous.
    // So in_stride and out_stride are both 1.

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardswish_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        in_stride,
        out_stride,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("hardswish_forward", &hardswish_forward,
          "Hardswish activation forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
