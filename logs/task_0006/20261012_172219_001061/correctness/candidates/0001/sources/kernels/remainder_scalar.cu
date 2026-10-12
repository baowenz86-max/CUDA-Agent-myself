#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

__global__ void remainder_scalar_kernel(
    const bool* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t divisor,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = idx; i < total_elements; i += stride) {
        int64_t value = (int64_t)input[i];
        // Python-like modulo for non-negative divisor (here 2) but avoid division by zero
        if (divisor == 0) {
            output[i] = 0; // mimic torch? Actually torch.remainder with zero gives NaN for float, but inputs are ints, undefined. We'll set 0.
        } else {
            int64_t rem = value % divisor;
            // Adjust for Python semantics if divisor positive: rem is non-negative
            if (rem < 0) rem += divisor;
            output[i] = rem;
        }
    }
}

torch::Tensor remainder_scalar_forward(torch::Tensor input, int64_t divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBool, "Only bool input is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt64));

    long total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    remainder_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<bool>(),
        output.data_ptr<int64_t>(),
        divisor,
        total_elements
    );

    return output;
}

void register_remainder_scalar(py::module& m) {
    m.def("remainder_scalar_forward", &remainder_scalar_forward,
          "Remainder scalar forward",
          py::arg("input"), py::arg("divisor"));
}

REGISTER_BINDING(remainder_scalar, register_remainder_scalar);
