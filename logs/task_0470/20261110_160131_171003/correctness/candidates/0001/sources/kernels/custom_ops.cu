#include <cuda_runtime.h>

// This model requires only CPU operations (device is 'cpu').
// The custom ops to_device_forward and abs_forward operate on CPU tensors,
// so no CUDA kernels are necessary in this .cu file.
// Their implementations will be provided in the C++ binding stage,
// as .cu files are restricted to CUDA kernels and launchers.
