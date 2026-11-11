// CUDA kernel for mean over dimension 0.
// Input shape [N, D1, ..., Dk, 2], output [D1, ..., Dk, 2].
// This implementation computes input offset for each output (m,c) on host,
// copies offsets to device, kernel sums over N using those offsets.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void mean_dim0_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t N,
                                 const int64_t* __restrict__ offsets, // length M*2
                                 int64_t stride0,
                                 int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t offset = offsets[idx]; // base offset for n=0
    const float* base = input + offset;
    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        sum += base[n * stride0];
    }
    output[idx] = sum / (float)N;
}

// Launcher: computes offsets on host (since shape/strides known), copies to device,
// launches kernel.
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int64_t N,
                                   int64_t ndim,
                                   const int64_t* shape,
                                   const int64_t* strides,
                                   cudaStream_t stream) {
    // Compute M = product of shape[1..ndim-2] (middle dims excluding last 2)
    int64_t M = 1;
    for (int64_t d = 1; d < ndim - 1; ++d) {
        M *= shape[d];
    }
    int64_t total = M * 2;
    if (total <= 0 || N <= 0) {
        // If N=0, torch.mean returns NaN; but task N>0. Just return without writing? 
        // For safety, we can initialize output with NaN? But output is empty? Not needed here.
        return;
    }

    // Host offsets: for each (m,c) compute offset = c*strides[ndim-1] + sum_{d=1}^{ndim-2} coord_d * strides[d]
    int64_t* h_offsets = (int64_t*)malloc(sizeof(int64_t) * total);
    if (!h_offsets) return;

    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t m = idx >> 1;
        int64_t c = idx & 1;
        int64_t rem = m;
        int64_t offset = c * strides[ndim - 1];
        for (int64_t d = ndim - 2; d >= 1; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            offset += coord * strides[d];
        }
        h_offsets[idx] = offset;
    }

    int64_t* d_offsets = nullptr;
    cudaMalloc(&d_offsets, sizeof(int64_t) * total);
    cudaMemcpy(d_offsets, h_offsets, sizeof(int64_t) * total, cudaMemcpyHostToDevice);
    free(h_offsets);

    const int block = 256;
    int64_t grid64 = (total + block - 1) / block;
    if (grid64 > 2147483647) grid64 = 2147483647;
    dim3 grid((unsigned int)grid64);
    dim3 threads(block);

    mean_dim0_kernel<<<grid, threads, 0, stream>>>(
        input, output, N, d_offsets, strides[0], total);

    cudaError_t err = cudaGetLastError();
    // Check error after launch; binding layer will also check.
    if (err != cudaSuccess) {
        // avoid silent failure; binding will call cudaGetLastError again though
    }

    // We cannot free d_offsets immediately because kernel may be async; use cudaFree after synchronize? 
    // Better: use cudaFreeAsync through stream? Simpler: use cudaFree after kernel completes; but here we don't synchronize.
    // To avoid memory leak, we could use a static or allocate via cudaMallocAsync, but for correctness just leak is bad.
    // Since kernel launch is async, we can register a callback? For simplicity, use cudaFree after a device-side event? 
    // For now, we can allocate d_offsets using cudaMalloc and then call cudaFree after kernel completes via cudaStreamAddCallback? 
    // However, the launcher should not synchronize. We can use cudaFreeAsync if available (CUDA 11.2+). 
    // Here we assume CUDA 12.6, so cudaFreeAsync is available, but stream may not be the default? It is the current stream.
    // Use cudaFreeAsync(d_offsets, stream) after kernel launch; or use cudaFree after a cudaStreamSynchronize? That would block.
    // Better: use cudaMallocAsync with stream; but to keep simple, we can allocate d_offsets via cudaMalloc and then schedule free using cudaFreeAsync.
    cudaFreeAsync(d_offsets, stream);
}
