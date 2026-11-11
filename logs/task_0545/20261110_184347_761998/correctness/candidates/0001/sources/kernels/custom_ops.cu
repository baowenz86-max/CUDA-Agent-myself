#include <cuda_runtime.h>

// Fused kernel: computes max over dim=1 (channels) with keepdim=True,
// then converts the result to complex64 via view_as_complex semantics.
// Input x: [batch, channels, height, width, 2] float32, contiguous
//   channels = 2 (always, per contract)
// Output complex_out: [batch, height, width] complex64 (stored as float2)
// Output indices: [batch, 1, height, width, 2] int64, each element is 0 or 1
//   indicating which channel contributed the max for that real/imag component.
__global__ void custom_max_view_as_complex_kernel(
    const float* __restrict__ x,
    float2* __restrict__ complex_out,
    long long* __restrict__ indices,
    int batch,
    int height,
    int width,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * height * width;
    if (idx >= total) return;

    // Decode flat index to (b, h, w)
    int w = idx % width;
    int tmp = idx / width;
    int h = tmp % height;
    int b = tmp / height;

    // Input layout: x[b][c][h][w][0/1]
    // Strides:
    //   b: channels * height * width * 2
    //   c: height * width * 2
    //   h: width * 2
    //   w: 2
    //   last dim: 1
    // Since input is contiguous per contract.
    const float* base = x + ((long long)b * channels * height * width * 2 +
                             (long long)h * width * 2 +
                             (long long)w * 2);

    // For channels=2, compare directly.
    // channel 0 values:
    float val0_real = base[0];
    float val0_imag = base[1];
    // channel 1 values (stride channels*height*width*2)
    const float* base1 = base + (long long)height * width * 2; // c=1 stride
    float val1_real = base1[0];
    float val1_imag = base1[1];

    // Determine max for real and imag independently.
    // For tie-breaking, torch.max returns the first index (channel 0), so use > not >=.
    int idx_real = (val1_real > val0_real) ? 1 : 0;
    int idx_imag = (val1_imag > val0_imag) ? 1 : 0;
    float max_real = (idx_real == 1) ? val1_real : val0_real;
    float max_imag = (idx_imag == 1) ? val1_imag : val0_imag;

    // Write complex output (real, imag)
    float2 val;
    val.x = max_real;
    val.y = max_imag;
    complex_out[idx] = val;

    // Write indices output: shape [batch,1,height,width,2]
    // The index tensor has the same spatial strides but with an extra dim of size 1.
    // Flattened index in indices: ((b*1 + 0)*height*width + h*width + w) * 2 + component
    long long indices_flat = ((long long)b * height * width + (long long)h * width + w) * 2;
    indices[indices_flat] = (long long)idx_real;
    indices[indices_flat + 1] = (long long)idx_imag;
}

extern "C" void custom_max_view_as_complex_launcher(
    const float* x,
    float2* complex_out,
    long long* indices,
    int batch,
    int height,
    int width,
    int channels,
    cudaStream_t stream)
{
    if (x == nullptr || complex_out == nullptr || indices == nullptr ||
        batch <= 0 || height <= 0 || width <= 0 || channels <= 0) {
        return;
    }
    // channels must be 2 per contract, but if not, we fallback to naive loop later.
    if (channels != 2) {
        // Not expected, but handle as error by not launching.
        // In real binding, should check and maybe throw. We just return.
        return;
    }

    int total = batch * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    custom_max_view_as_complex_kernel<<<blocks, threads, 0, stream>>>(
        x, complex_out, indices, batch, height, width, channels);

    // Optional: check for launch errors (binding may also check)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) return;
}
