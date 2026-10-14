// Not part of the extension. Compile this on its own to look at the generated PTX:
//   nvcc -arch=sm_80 -std=c++17 -ptx csrc/ptx_check.cu -o results/kernels.ptx
//   grep -o "ld\.global[.a-z0-9]*\|st\.global[.a-z0-9]*\|shfl[.a-z0-9]*\|\.entry [A-Za-z0-9_]*" results/kernels.ptx | uniq -c
// v0 should show 16-bit loads (ld.global.nc.u16), v1 should show 128-bit ones (ld.global.nc.v4.u32)
// plus shfl.sync.bfly for the warp reduction.
#include "rmsnorm_kernels.cuh"

template __global__ void rmsnorm_v0_kernel<__half>(const __half*, const __half*, __half*, int, float);
template __global__ void rmsnorm_v1_kernel<__half>(const __half*, const __half*, __half*, int, float);
template __global__ void fused_add_rmsnorm_v1_kernel<__half>(__half*, __half*, const __half*, int, float);
