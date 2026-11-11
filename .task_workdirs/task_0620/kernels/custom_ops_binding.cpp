#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers defined in custom_ops.cu
extern "C" void hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void avg_pool3d_launcher(
    float* output,
    const float* input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t outD, int64_t outH, int64_t outW,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t* input_strides,
    int64_t* output_strides,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// hardtanh wrapper
torch::Tensor hardtanh(torch::Tensor input, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    // For elementwise operation, any stride layout is fine; no need to contiguous.
    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    if (numel > 0) {
        hardtanh_launcher(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            numel,
            (float)min_val,
            (float)max_val,
            get_stream()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

// avg_pool3d wrapper
torch::Tensor avg_pool3d(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "invalid pooling parameters");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    int64_t outD = (D + 2 * padding - kernel_size) / stride + 1;
    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "kernel size too large for input");

    auto output = torch::empty({N, C, outD, outH, outW}, input.options());

    // Launcher expects raw pointers and strides arrays on device.
    // Since input may be non-contiguous, we create a contiguous copy if needed.
    torch::Tensor input_contig = input.contiguous();
    // For output, we allocate contiguous by torch::empty, so its strides are contiguous.
    // We need to pass strides arrays to launcher; allocate on host and copy to device,
    // or use static arrays and transfer? The launcher takes int64_t* pointers,
    // so we need device memory for strides. Here we allocate small CUDA tensors for strides.
    // Alternatively, we can simplify the launcher to assume contiguous layout,
    // but the existing launcher expects strides. We'll provide strides arrays.
    auto input_strides = torch::empty({5}, input.options().dtype(torch::kInt64));
    auto output_strides = torch::empty({5}, input.options().dtype(torch::kInt64));
    int64_t input_stride_arr[5] = {
        input_contig.stride(0),
        input_contig.stride(1),
        input_contig.stride(2),
        input_contig.stride(3),
        input_contig.stride(4)
    };
    int64_t output_stride_arr[5] = {
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4)
    };
    // Copy host arrays to device tensors (synchronous but small)
    cudaMemcpy(input_strides.data_ptr<int64_t>(), input_stride_arr, 5 * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_strides.data_ptr<int64_t>(), output_stride_arr, 5 * sizeof(int64_t), cudaMemcpyHostToDevice);

    avg_pool3d_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        N, C, D, H, W,
        outD, outH, outW,
        kernel_size, stride, padding,
        input_strides.data_ptr<int64_t>(),
        output_strides.data_ptr<int64_t>(),
        get_stream()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Strides tensors are freed after function returns; fine because kernel is asynchronous
    // and uses the pointers before completion? Actually cudaMemcpy is synchronous,
    // and the kernel launch is queued after the memcopies on the same stream,
    // so memory stays allocated until stream sync. Since we don't synchronize,
    // the tensors' memory may be freed early if we don't retain them. To be safe,
    // we create them as local variables and they will be freed at end of scope,
    // but the kernel may still be using the memory. Need to keep them alive until kernel finishes.
    // We can use cudaStreamSynchronize or use a deleter. Simpler: synchronize at end.
    cudaStreamSynchronize(get_stream());
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh", &hardtanh, "Hardtanh elementwise clamp",
          py::arg("input"), py::arg("min_val") = -1.0, py::arg("max_val") = 1.0);
    m.def("avg_pool3d", &avg_pool3d, "3D average pooling",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
