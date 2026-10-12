#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        int ow = linear_idx % out_w;
        int oh = (linear_idx / out_w) % out_h;
        long plane = linear_idx / (long)(out_w * out_h);
        int c = plane % channels;
        int b = plane / channels;

        int iw = ow - pad_left;
        int ih = oh - pad_top;

        // reflect indices
        if (iw < 0) iw = -iw;
        if (iw >= in_w) iw = 2 * in_w - 2 - iw;
        if (ih < 0) ih = -ih;
        if (ih >= in_h) ih = 2 * in_h - 2 - ih;

        long input_idx = ((long)b * channels + c) * in_h * in_w + (long)ih * in_w + iw;
        output[linear_idx] = input[input_idx];
    }
}

torch::Tensor reflection_pad2d_forward(torch::Tensor input, std::vector<int64_t> padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(padding.size() == 4, "Padding must be a 4-element vector (left, right, top, bottom)");

    int64_t pad_left = padding[0];
    int64_t pad_right = padding[1];
    int64_t pad_top = padding[2];
    int64_t pad_bottom = padding[3];

    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;

    TORCH_CHECK(out_h >= 1 && out_w >= 1, "Output size must be positive");

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    long total_elements = (long)batch * channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        pad_left, pad_right, pad_top, pad_bottom,
        total_elements
    );

    return output;
}

void register_reflection_pad2d(py::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection padding 2D forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
