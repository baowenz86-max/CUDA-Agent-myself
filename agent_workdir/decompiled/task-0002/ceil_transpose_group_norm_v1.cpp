#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

extern void launch_fused_group_norm_kernel(const float* d_input,
                                           float* d_output,
                                           int n,
                                           int c,
                                           int h,
                                           int w,
                                           int groups,
                                           cudaStream_t stream = 0);

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

int main() {
    const int n = 1;
    const int c = 8;
    const int h = 2;
    const int w = 2;
    const int groups = 2;

    const size_t total = static_cast<size_t>(n) * c * h * w;
    std::vector<float> host_input(total);
    std::vector<float> host_output(total);

    for (size_t i = 0; i < total; ++i) {
        host_input[i] = static_cast<float>((i % 7) - 3);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    checkCuda(cudaMalloc(&d_input, total * sizeof(float)), "cudaMalloc d_input");
    checkCuda(cudaMalloc(&d_output, total * sizeof(float)), "cudaMalloc d_output");

    checkCuda(cudaMemcpy(d_input, host_input.data(), total * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy input to device");

    launch_fused_group_norm_kernel(d_input, d_output, n, c, h, w, groups);
    checkCuda(cudaGetLastError(), "launch_fused_group_norm_kernel");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaMemcpy(host_output.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy output to host");

    std::cout << "Kernel output sample:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(8, total); ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
