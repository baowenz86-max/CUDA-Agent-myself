#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// Kernel 1: AddMM forward
// Computes C = bias + A * B
// A is [M, K], B is [K, N] (i.e., self.weight with shape [hidden_size, input_size],
// but for the multiplication we treat it as [K, N] where K=input_size, N=hidden_size).
// Output C is [M, N]
// bias is [N]
// All tensors are float32, contiguous.
// -----------------------------------------------------------------------------
__global__ void addmm_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             const float* __restrict__ bias,
                             float* __restrict__ C,
                             int M, int N, int K) {
    int total = M * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / N;
        int col = idx % N;
        float sum = bias[col];
        for (int k = 0; k < K; ++k) {
            // B is [K, N], so element B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }
        C[idx] = sum;
    }
}

// Launcher for addmm
// Inputs: bias [N], A [M,K], B [K,N]; Output: C [M,N]
extern "C" void addmm_forward_launcher(const float* bias,
                                       const float* A,
                                       const float* B,
                                       float* C,
                                       int M, int N, int K,
                                       cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int total = M * N;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    addmm_kernel<<<blocks, threads, 0, stream>>>(A, B, bias, C, M, N, K);
}

// -----------------------------------------------------------------------------
// Kernel 2: Unsqueeze forward (adds a dimension at specified position, but here only dim=0 is used)
// Input shape: [M, N], output shape: [1, M, N]. Data is copied.
// -----------------------------------------------------------------------------
__global__ void unsqueeze_copy_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < total_elements; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void unsqueeze_forward_launcher(const float* input,
                                           float* output,
                                           int total_elements,
                                           int dim,
                                           cudaStream_t stream) {
    if (total_elements <= 0) return;
    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    unsqueeze_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// -----------------------------------------------------------------------------
// Kernel 3: Linear interpolation for 3D input [1, M, N] with scale_factor=2
// Output shape [1, M, 2*N]
// Align_corners=False: out_x = (in_x + 0.5) * scale - 0.5; clamp to [0, N-1]
// scale = input_width / output_width
// -----------------------------------------------------------------------------
__global__ void interpolate_linear_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int input_height,   // M
                                          int input_width,    // N
                                          int output_width,   // out_N
                                          float scale) {
    int total = input_height * output_width; // batch=1
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int m = idx / output_width;
        int out_x = idx % output_width;
        float src_x_f = (static_cast<float>(out_x) + 0.5f) * scale - 0.5f;
        src_x_f = fmaxf(0.0f, fminf(src_x_f, static_cast<float>(input_width - 1)));
        int x0 = static_cast<int>(floorf(src_x_f));
        int x1 = min(x0 + 1, input_width - 1);
        float dx = src_x_f - static_cast<float>(x0);
        const float* row_ptr = input + m * input_width;
        float val = row_ptr[x0] * (1.0f - dx) + row_ptr[x1] * dx;
        output[idx] = val;
    }
}

extern "C" void interpolate_linear_forward_launcher(const float* input,
                                                    float* output,
                                                    int input_height,
                                                    int input_width,
                                                    int output_width,
                                                    float scale_factor,
                                                    cudaStream_t stream) {
    if (input_height <= 0 || input_width <= 0 || output_width <= 0) return;
    int total = input_height * output_width;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    float scale = static_cast<float>(input_width) / static_cast<float>(output_width);
    interpolate_linear_kernel<<<blocks, threads, 0, stream>>>(
        input, output, input_height, input_width, output_width, scale);
}
