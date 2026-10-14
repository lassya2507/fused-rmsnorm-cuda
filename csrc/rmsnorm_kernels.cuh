// RMSNorm and fused add + RMSNorm kernels.
//
// Layout: x is [num_tokens, hidden], row-major. One thread block handles one row
// (one token), so gridDim.x = num_tokens.
//
// RMSNorm:            out = x / sqrt(mean(x^2) + eps) * w
// Fused add + RMSNorm: residual = residual + x;  x = rmsnorm(residual) * w   (both in place)
//
// All math is done in fp32 and only the loads/stores are fp16/bf16.

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// 8 x 16-bit values = 16 bytes. With 16-byte alignment the compiler can move this
// with a single 128-bit load/store (ld.global.v4 / st.global.v4) instead of 8 small ones.
constexpr int kVec = 8;
template <typename T>
struct alignas(16) Vec8 {
  T data[kVec];
};

// Sum a value across the 32 threads of a warp using register shuffles (no shared memory).
// After 5 xor steps (16, 8, 4, 2, 1) every lane holds the full warp sum.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

// Sum a value across the whole block. Two levels:
//   1) each warp reduces itself with shuffles
//   2) warp 0 reduces the per-warp partial sums
// Needs blockDim.x to be a multiple of 32 (at most 1024 -> at most 32 warps).
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  __shared__ float block_total;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0) {
    float partial = (lane < num_warps) ? warp_sums[lane] : 0.f;
    partial = warp_reduce_sum(partial);
    if (lane == 0) block_total = partial;
  }
  __syncthreads();
  return block_total;
}

// ---------------------------------------------------------------------------
// v0: the straightforward version
//   - scalar (2-byte) loads
//   - classic shared-memory tree reduction, one __syncthreads per level
//   - blockDim.x must be a power of two
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_v0_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                  T* __restrict__ out, int hidden, float eps) {
  extern __shared__ float smem[];  // blockDim.x floats
  const T* x_row = x + static_cast<size_t>(blockIdx.x) * hidden;
  T* out_row = out + static_cast<size_t>(blockIdx.x) * hidden;

  // pass 1: each thread accumulates its share of sum(x^2)
  float sum_sq = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = to_float(x_row[i]);
    sum_sq += v * v;
  }

  // tree reduction in shared memory: 256 -> 128 -> ... -> 1
  smem[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(smem[0] / hidden + eps);

  // pass 2: read x again (hopefully from L2 now), scale, write out
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    out_row[i] = from_float<T>(to_float(x_row[i]) * inv_rms * to_float(w[i]));
  }
}

// ---------------------------------------------------------------------------
// v1: same algorithm, two changes
//   - 128-bit vectorized loads/stores (8 elements per memory instruction)
//   - warp-shuffle reduction instead of the shared-memory tree
// Requires hidden % 8 == 0 and 16-byte aligned pointers (checked on the host).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_v1_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                  T* __restrict__ out, int hidden, float eps) {
  using V = Vec8<T>;
  const int num_vecs = hidden / kVec;
  const V* x_row = reinterpret_cast<const V*>(x + static_cast<size_t>(blockIdx.x) * hidden);
  const V* w_vec = reinterpret_cast<const V*>(w);
  V* out_row = reinterpret_cast<V*>(out + static_cast<size_t>(blockIdx.x) * hidden);

  float sum_sq = 0.f;
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    V v = x_row[i];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float f = to_float(v.data[j]);
      sum_sq += f * f;
    }
  }

  const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / hidden + eps);

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    V v = x_row[i];
    V wv = w_vec[i];
    V o;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      o.data[j] = from_float<T>(to_float(v.data[j]) * inv_rms * to_float(wv.data[j]));
    }
    out_row[i] = o;
  }
}

// ---------------------------------------------------------------------------
// fused add + RMSNorm (same semantics as flashinfer.norm.fused_add_rmsnorm)
//   residual <- residual + input
//   input    <- rmsnorm(residual) * w
// The sum is rounded to T before it is stored and before it is squared, so the
// norm is computed on exactly the residual value that later layers will see.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_add_rmsnorm_v1_kernel(T* __restrict__ input, T* __restrict__ residual,
                                            const T* __restrict__ w, int hidden, float eps) {
  using V = Vec8<T>;
  const int num_vecs = hidden / kVec;
  V* in_row = reinterpret_cast<V*>(input + static_cast<size_t>(blockIdx.x) * hidden);
  V* res_row = reinterpret_cast<V*>(residual + static_cast<size_t>(blockIdx.x) * hidden);
  const V* w_vec = reinterpret_cast<const V*>(w);

  // pass 1: add, store the new residual, accumulate sum of squares
  float sum_sq = 0.f;
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    V a = in_row[i];
    V r = res_row[i];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      r.data[j] = from_float<T>(to_float(a.data[j]) + to_float(r.data[j]));
      float f = to_float(r.data[j]);
      sum_sq += f * f;
    }
    res_row[i] = r;
  }

  const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / hidden + eps);

  // pass 2: every thread re-reads only the residual chunks it wrote itself,
  // so no extra synchronization is needed for those writes to be visible.
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    V r = res_row[i];
    V wv = w_vec[i];
    V o;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      o.data[j] = from_float<T>(to_float(r.data[j]) * inv_rms * to_float(wv.data[j]));
    }
    in_row[i] = o;
  }
}
