#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_reflection_pad2d_launcher(
    float* output,
    const float* input,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream);

extern "C" void custom_roll_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t ndim,
    int64_t* dims_device,
    int64_t* shifts_device,
    int64_t dims_size,
    int64_t* shape_device,
    int64_t* in_strides_device,
    int64_t* out_strides_device,
    cudaStream_t stream);

static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor custom_reflection_pad2d(torch::Tensor x, torch::Tensor pad) {
    TORCH_CHECK(x.is_cuda(), "custom_reflection_pad2d: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_reflection_pad2d: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_reflection_pad2d: x must be 4D");
    TORCH_CHECK(pad.device().is_cpu() && pad.dtype() == torch::kInt64 && pad.numel() == 4,
                "custom_reflection_pad2d: pad must be a CPU int64 tensor of size 4");

    // Extract pad values (left, right, top, bottom)
    auto pad_acc = pad.accessor<int64_t, 1>();
    int64_t pad_left = pad_acc[0];
    int64_t pad_right = pad_acc[1];
    int64_t pad_top = pad_acc[2];
    int64_t pad_bottom = pad_acc[3];

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    // Create contiguous output
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get input strides (original or contiguous)
    auto x_contig = x.contiguous();
    int64_t in_stride_n = x_contig.stride(0);
    int64_t in_stride_c = x_contig.stride(1);
    int64_t in_stride_h = x_contig.stride(2);
    int64_t in_stride_w = x_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_reflection_pad2d_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        pad_left, pad_right, pad_top, pad_bottom,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        stream);
    check_cuda_error("custom_reflection_pad2d");
    return output;
}

torch::Tensor custom_roll(torch::Tensor x, torch::Tensor shifts, torch::Tensor dims) {
    TORCH_CHECK(x.is_cuda(), "custom_roll: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_roll: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_roll: x must be 4D");
    TORCH_CHECK(shifts.device().is_cpu() && shifts.dtype() == torch::kInt64,
                "custom_roll: shifts must be a CPU int64 tensor");
    TORCH_CHECK(dims.device().is_cpu() && dims.dtype() == torch::kInt64,
                "custom_roll: dims must be a CPU int64 tensor");
    TORCH_CHECK(shifts.numel() == dims.numel(), "custom_roll: shifts and dims must have same size");

    int64_t dims_size = shifts.numel();
    TORCH_CHECK(dims_size == 2, "custom_roll: only two dims supported for this task");

    auto shifts_acc = shifts.accessor<int64_t, 1>();
    auto dims_acc = dims.accessor<int64_t, 1>();

    auto output = torch::empty_like(x);

    // Create GPU copies of metadata
    std::vector<int64_t> shape_host(4);
    std::vector<int64_t> in_strides_host(4);
    std::vector<int64_t> out_strides_host(4);
    for (int d = 0; d < 4; ++d) {
        shape_host[d] = x.size(d);
        in_strides_host[d] = x.stride(d);
        out_strides_host[d] = output.stride(d);
    }

    int64_t* shape_device;
    int64_t* in_strides_device;
    int64_t* out_strides_device;
    int64_t* dims_device;
    int64_t* shifts_device;

    cudaMalloc(&shape_device, 4 * sizeof(int64_t));
    cudaMalloc(&in_strides_device, 4 * sizeof(int64_t));
    cudaMalloc(&out_strides_device, 4 * sizeof(int64_t));
    cudaMalloc(&dims_device, dims_size * sizeof(int64_t));
    cudaMalloc(&shifts_device, dims_size * sizeof(int64_t));

    cudaMemcpy(shape_device, shape_host.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(in_strides_device, in_strides_host.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_strides_device, out_strides_host.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Copy dims and shifts to device
    std::vector<int64_t> dims_host(dims_size);
    std::vector<int64_t> shifts_host(dims_size);
    for (int64_t i = 0; i < dims_size; ++i) {
        dims_host[i] = dims_acc[i];
        shifts_host[i] = shifts_acc[i];
    }
    cudaMemcpy(dims_device, dims_host.data(), dims_size * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(shifts_device, shifts_host.data(), dims_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_roll_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        4,
        dims_device,
        shifts_device,
        dims_size,
        shape_device,
        in_strides_device,
        out_strides_device,
        stream);
    check_cuda_error("custom_roll");

    cudaFree(shape_device);
    cudaFree(in_strides_device);
    cudaFree(out_strides_device);
    cudaFree(dims_device);
    cudaFree(shifts_device);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_reflection_pad2d", &custom_reflection_pad2d,
          "Apply ReflectionPad2d with given padding",
          py::arg("x"), py::arg("pad"));
    m.def("custom_roll", &custom_roll,
          "Perform torch.roll on specified dims",
          py::arg("x"), py::arg("shifts"), py::arg("dims"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
