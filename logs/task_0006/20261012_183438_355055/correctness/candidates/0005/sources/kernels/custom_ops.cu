#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../binding_registry.h"

namespace py = pybind11;

__global__ void fused_pad_ge_remainder_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int out_H, int out_W,
    long total_elements)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;

    for (long i = idx; i < total_elements; i += stride) {
        int ow = i % out_W;
        long tmp = i / out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int c = tmp % C;
        int n = tmp / C;

        int iw;
        if (ow < pad_left) {
            iw = pad_left - ow;
        } else if (ow >= pad_left + W) {
            iw = 2 * W + pad_left - ow - 2;
        } else {
            iw = ow - pad_left;
        }

        int ih;
        if (oh < pad_top) {
            ih = pad_top - oh;
        } else if (oh >= pad_top + H) {
            ih = 2 * H + pad_top - oh - 2;
        } else {
            ih = oh - pad_top;
        }

        float val = input[((long)n * C + c) * H * W + (long)ih * W + iw];

        float ge_result = (val >= 0.0f) ? 1.0f : 0.0f;

        int64_t rem_result = (int64_t)(ge_result);

        output[i] = rem_result;
    }
}

torch::Tensor custom_ops_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;

    auto output = torch::empty({N, C, out_H, out_W}, input.options().dtype(torch::kInt64));

    long total_elements = (long)N * C * out_H * out_W;
    if (total_elements == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    fused_pad_ge_remainder_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        out_H, out_W,
        total_elements
    );

    return output;
}

void register_custom_ops(py::module& m) {
    m.def("custom_ops_forward", &custom_ops_forward,
          "Fused reflection pad + ge + remainder",
          py::arg("input"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
