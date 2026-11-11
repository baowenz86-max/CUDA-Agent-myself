#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_transpose_launcher(
    float* out,
    const float* in,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
    int64_t total,
    cudaStream_t stream);

extern "C" void custom_selu_launcher(
    float* out,
    const float* in,
    int64_t total,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

// Helper to convert dims to positive indices
static inline int64_t normalize_dim(int64_t dim, int64_t ndim) {
    return dim < 0 ? dim + ndim : dim;
}

// custom_transpose wrapper
torch::Tensor custom_transpose(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "custom_transpose: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_transpose: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_transpose: expected 4D tensor");

    dim0 = normalize_dim(dim0, x.dim());
    dim1 = normalize_dim(dim1, x.dim());
    TORCH_CHECK(dim0 >= 0 && dim0 < 4 && dim1 >= 0 && dim1 < 4, "custom_transpose: dims out of range");
    TORCH_CHECK(dim0 != dim1, "custom_transpose: dims must be different");

    // For this specific case, dims are 1 and 2; but we keep general for safety.
    // We'll implement for 4D only and assume swap of exactly dims 1 and 2 if not that.
    // The kernel expects N,C,H,W and swaps dims 1 and 2. So enforce that.
    TORCH_CHECK(dim0 == 1 && dim1 == 2, "custom_transpose: only support swapping dims 1 and 2");

    // Input shape: [N, C, H, W]
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Input strides
    int64_t in_stride0 = x.stride(0);
    int64_t in_stride1 = x.stride(1);
    int64_t in_stride2 = x.stride(2);
    // in_stride3 is not needed (implicit 1). For safety, we don't use it.

    // Output shape: [N, H, C, W]
    // We allocate output with the desired strides matching torch.transpose result.
    // torch.transpose returns a view with strides swapped, but for custom op we need a new tensor
    // that has data in the transposed layout. We can allocate a contiguous tensor of shape [N,H,C,W]
    // and then set its strides? Actually easiest: allocate empty with shape [N,H,C,W] and then
    // use as_strided to set the strides to [out_stride0, out_stride1, out_stride2, 1]? No, the
    // output should have strides [N*H*C*W? Actually reference stride is [4194304, 256, 131072, 1]
    // which is exactly the transposed view of the contiguous input. If we allocate a new tensor
    // with shape [N,H,C,W] using torch::empty, it will be contiguous with strides [H*C*W, C*W, W, 1].
    // That's different from [N*C*W, W, C*W, 1]? Wait: For shape [128,512,32,256], contiguous strides:
    // [512*32*256, 32*256, 256, 1] = [4194304, 8192, 256, 1]. But reference strides are
    // [4194304, 256, 131072, 1]. That's a non-contiguous view of the original contiguous input with
    // dims swapped. So the reference output is a view, not a new contiguous tensor. However, we
    // can still return a tensor with that exact stride by using as_strided, but then we need the
    // underlying storage to be large enough. The launcher writes to out pointer assuming the
    // provided strides. So we could allocate a contiguous buffer of size total, then create a
    // view with the transposed strides using as_strided. But then the data layout in memory
    // must match the indexing in kernel. The kernel writes to out[dst_idx] where dst_idx =
    // n*out_stride0 + h*out_stride1 + c*out_stride2 + w. So we need to allocate a buffer of at
    // least the maximum index + 1. For non-overlapping contiguous-like strides, the maximum
    // index is (N-1)*out_stride0 + (H-1)*out_stride1 + (C-1)*out_stride2 + (W-1). That's
    // (127)*4194304 + (511)*256 + (31)*131072 + 255 = 532676608 + 130816 + 4063232 + 255 =
    // 536870911 (total-1). So a buffer of size total is enough. Then we can return a tensor
    // with storage offset 0 and strides as specified. But the buffer itself is contiguous in
    // memory, and the kernel will write to indices up to total-1. That works.
    // However, if we allocate a contiguous tensor of shape [N,H,C,W], its strides differ.
    // We need to return a tensor with the correct strides. We can create an empty tensor with
    // size [N,H,C,W] (contiguous), then call as_strided to return a view with the desired
    // strides. But then the storage size must be at least the largest index + 1. The contiguous
    // storage size is total, so it is sufficient. So we can allocate a contiguous tensor of
    // shape [N,H,C,W] and then return its as_strided with the desired strides.
    // Alternatively, we can allocate a 1D buffer of size total, then as_strided it.
    // Simpler: allocate output = torch::empty({N,H,C,W}, x.options()); // contiguous
    // Then output = output.as_strided({N,H,C,W}, {out_stride0, out_stride1, out_stride2, 1});
    // But that returns a view of the same underlying storage; the kernel will write using the
    // launcher with out_stride provided, which we set to the desired strides. So we need to
    // compute out_stride values: For transposed view, out_stride0 = in_stride0 (4194304),
    // out_stride1 = in_stride2 (256), out_stride2 = in_stride1 (131072), out_stride3 = 1.
    // So we can compute those from input strides.

    int64_t out_stride0 = in_stride0;
    int64_t out_stride1 = in_stride2;
    int64_t out_stride2 = in_stride1;

    // Allocate contiguous storage of total elements.
    auto output_storage = torch::empty({N*H*C*W}, x.options());
    // Create view with desired shape and strides.
    auto output = output_storage.as_strided({N,H,C,W}, {out_stride0, out_stride1, out_stride2, 1});

    int64_t total = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_transpose_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        in_stride0, in_stride1, in_stride2,
        out_stride0, out_stride1, out_stride2,
        total,
        stream
    );

    return output;
}

// custom_selu wrapper
torch::Tensor custom_selu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_selu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_selu: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_selu: expected 4D tensor");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Input strides
    int64_t in_stride0 = x.stride(0);
    int64_t in_stride1 = x.stride(1);
    int64_t in_stride2 = x.stride(2);

    // Output should have the same shape and strides as input (per contract).
    // We can allocate a new tensor with the same strides as input, but we need to ensure
    // the storage is large enough. Since input may be non-contiguous, we'll allocate a
    // contiguous buffer of numel elements and create a view with the same strides as input.
    int64_t total = x.numel();
    auto output_storage = torch::empty({total}, x.options());
    auto output = output_storage.as_strided({N,C,H,W}, {in_stride0, in_stride1, in_stride2, 1});

    // Output strides (same as input)
    int64_t out_stride0 = in_stride0;
    int64_t out_stride1 = in_stride1;
    int64_t out_stride2 = in_stride2;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_selu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        total,
        in_stride0, in_stride1, in_stride2,
        out_stride0, out_stride1, out_stride2,
        N, C, H, W,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_transpose", &custom_transpose,
          "Transpose dimensions 1 and 2 of a 4D tensor",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"));
    m.def("custom_selu", &custom_selu,
          "Apply SELU activation elementwise",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
