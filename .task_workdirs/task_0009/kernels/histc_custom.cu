#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>

// Kernel: compute histogram with linear scaling of values to bins
// Each thread processes one element of the input tensor (contiguous required).
// Bins are counted in global memory using atomicAdd, which is correct but may be slow for high contention.
// For better performance, a two-stage approach with privatized histograms in shared memory could be used,
// but this is sufficient for the required tensor sizes (typically ~25M elements, 20 bins).
__global__ void histc_linear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int bins,
    float min_val,
    float max_val,
    float bin_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = input[idx];
    // Clamp to [min_val, max_val] as per torch.clamp
    val = fminf(fmaxf(val, min_val), max_val);

    // Scale linearly: index = floor((val - min_val) / bin_width)
    int bin = (int)floorf((val - min_val) / bin_width);

    // Ensure bin is in [0, bins-1] (due to floating point and max_val case)
    if (bin < 0) bin = 0;
    if (bin >= bins) bin = bins - 1;

    atomicAdd(&output[bin], 1.0f);
}

// Host launcher function with C linkage for binding
// Input: pointer to contiguous float tensor (input), pointer to output histogram (float*), num_elements, bins, min_val, max_val, stream
// Output: histogram counts (floats, for torch.float32 output)
extern "C" void histc_custom_launcher(
    const float* input,
    float* output,
    int num_elements,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    // Compute bin width assuming uniform spacing
    if (bins <= 0) return;
    float bin_width = (max_val - min_val) / bins;

    // Launch kernel with grid-stride loop? We simply launch enough blocks to cover all elements.
    // Use 256 threads per block, grid size = ceil(num_elements/256)
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    histc_linear_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, num_elements, bins, min_val, max_val, bin_width);
}

// PyTorch binding function that takes a tensor and returns histogram tensor
// This is a simple wrapper needed for Python API. The actual launcher is C-style.
// The binding file will include this .cu and call the launcher directly; however,
// we can also define a binding function here for convenience, but per instructions we should not generate .cpp.
// The user said not to modify .cpp, so this .cu must provide the launcher only.
// The binding file will call histc_custom_launcher.
// Provide a Python-visible function? The binding is in _binding.cpp; it will call the launcher.
// No additional code needed.
