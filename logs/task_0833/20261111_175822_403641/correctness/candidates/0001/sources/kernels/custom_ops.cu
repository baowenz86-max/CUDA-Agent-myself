#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void triu_2d_forward_kernel(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t diagonal,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w) {
  int64_t total = N * C * H * W;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
    int64_t w = idx % W;
    int64_t h = (idx / W) % H;
    int64_t c = (idx / (W * H)) % C;
    int64_t n = idx / (W * H * C);

    int64_t in_offset = n * input_stride_n + c * input_stride_c +
                        h * input_stride_h + w * input_stride_w;
    int64_t out_offset = n * output_stride_n + c * output_stride_c +
                         h * output_stride_h + w * output_stride_w;

    float val = 0.0f;
    if (w >= h + diagonal) {
      val = input[in_offset];
    }
    output[out_offset] = val;
  }
}

__global__ void batch_norm2d_eval_forward_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w) {
  int64_t total = N * C * H * W;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
    int64_t w = idx % W;
    int64_t h = (idx / W) % H;
    int64_t c = (idx / (W * H)) % C;
    int64_t n = idx / (W * H * C);

    int64_t in_offset = n * input_stride_n + c * input_stride_c +
                        h * input_stride_h + w * input_stride_w;
    int64_t out_offset = n * output_stride_n + c * output_stride_c +
                         h * output_stride_h + w * output_stride_w;

    float invstd = rsqrtf(static_cast<float>(running_var[c] + eps));
    float mean = running_mean[c];
    float gamma = weight ? weight[c] : 1.0f;
    float beta = bias ? bias[c] : 0.0f;

    float val = (input[in_offset] - mean) * invstd * gamma + beta;
    output[out_offset] = val;
  }
}

} // namespace

extern "C" {

void triu_2d_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t diagonal,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream) {
  int64_t total = N * C * H * W;
  if (total == 0) return;

  int threads = 256;
  int64_t blocks64 = (total + threads - 1) / threads;
  if (blocks64 > 2147483647) blocks64 = 2147483647;
  int blocks = static_cast<int>(blocks64);

  triu_2d_forward_kernel<<<blocks, threads, 0, stream>>>(
      output, input, N, C, H, W, diagonal,
      input_stride_n, input_stride_c, input_stride_h, input_stride_w,
      output_stride_n, output_stride_c, output_stride_h, output_stride_w);
}

void batch_norm2d_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream) {
  int64_t total = N * C * H * W;
  if (total == 0) return;

  int threads = 256;
  int64_t blocks64 = (total + threads - 1) / threads;
  if (blocks64 > 2147483647) blocks64 = 2147483647;
  int blocks = static_cast<int>(blocks64);

  batch_norm2d_eval_forward_kernel<<<blocks, threads, 0, stream>>>(
      output, input, weight, bias, running_mean, running_var, eps,
      N, C, H, W,
      input_stride_n, input_stride_c, input_stride_h, input_stride_w,
      output_stride_n, output_stride_c, output_stride_h, output_stride_w);
}

} // extern "C"
