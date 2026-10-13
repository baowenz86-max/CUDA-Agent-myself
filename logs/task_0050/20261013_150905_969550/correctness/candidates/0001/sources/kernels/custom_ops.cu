#include <cuda_runtime.h>

// The model consists solely of a Conv2d operation.
// Per the task contract, Conv2d is implemented via cuDNN in the host-side binding code (third stage).
// No custom CUDA kernels are required; this file exists to satisfy the requirement of a single .cu file.
