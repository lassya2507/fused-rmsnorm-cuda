// PyTorch bindings: input checks, dtype dispatch and kernel launch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

#include "rmsnorm_kernels.cuh"

namespace {

// block_size <= 0 means "pick for me": one 16-byte vector (8 values) per thread,
// capped at 1024 threads and rounded up to a whole number of warps.
// (This is the same rule FlashInfer uses: block = min(1024, hidden / vec_size).)
int64_t auto_block_size(int64_t hidden) {
  int64_t threads = std::min<int64_t>(1024, (hidden + kVec - 1) / kVec);
  return std::max<int64_t>(32, (threads + 31) / 32 * 32);
}

void check_common(const torch::Tensor& x, const torch::Tensor& w, int64_t block_size) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must be on the GPU");
  TORCH_CHECK(x.dim() == 2, "x must be [num_tokens, hidden]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(1), "w must be [hidden]");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w must have the same dtype");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "only fp16 and bf16 are supported");
  TORCH_CHECK(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0,
              "block_size must be a multiple of 32 in [32, 1024]");
}

void check_vectorizable(const torch::Tensor& t) {
  TORCH_CHECK(t.size(-1) % kVec == 0, "hidden must be a multiple of 8 for the vectorized kernel");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "pointer must be 16-byte aligned");
}

template <typename T>
void launch_rmsnorm(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                    double eps, int64_t version, int64_t block_size) {
  const int num_tokens = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  if (num_tokens == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const T*>(x.data_ptr());
  auto wp = reinterpret_cast<const T*>(w.data_ptr());
  auto op = reinterpret_cast<T*>(out.data_ptr());
  dim3 grid(num_tokens), block(block_size);

  if (version == 0) {
    size_t smem = block_size * sizeof(float);
    rmsnorm_v0_kernel<T><<<grid, block, smem, stream>>>(xp, wp, op, hidden, static_cast<float>(eps));
  } else {
    rmsnorm_v1_kernel<T><<<grid, block, 0, stream>>>(xp, wp, op, hidden, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps, int64_t version, int64_t block_size) {
  if (block_size <= 0) {
    TORCH_CHECK(version == 1, "auto block size is only for the v1 kernel");
    block_size = auto_block_size(x.size(-1));
  }
  check_common(x, w, block_size);
  TORCH_CHECK(version == 0 || version == 1, "version must be 0 or 1");
  if (version == 0) {
    TORCH_CHECK((block_size & (block_size - 1)) == 0, "v0 needs a power-of-two block_size");
  } else {
    check_vectorizable(x);
    check_vectorizable(w);
  }
  auto out = torch::empty_like(x);
  if (x.scalar_type() == at::kHalf) {
    launch_rmsnorm<__half>(x, w, out, eps, version, block_size);
  } else {
    launch_rmsnorm<__nv_bfloat16>(x, w, out, eps, version, block_size);
  }
  return out;
}

// In place, like flashinfer.norm.fused_add_rmsnorm: residual += input; input = rmsnorm(residual) * w
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor w, double eps,
                       int64_t block_size) {
  if (block_size <= 0) block_size = auto_block_size(input.size(-1));
  check_common(input, w, block_size);
  TORCH_CHECK(residual.sizes() == input.sizes() && residual.scalar_type() == input.scalar_type() &&
                  residual.is_contiguous() && residual.is_cuda(),
              "residual must match input");
  check_vectorizable(input);
  check_vectorizable(residual);
  check_vectorizable(w);

  const int num_tokens = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  if (num_tokens == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_tokens), block(block_size);

  if (input.scalar_type() == at::kHalf) {
    using T = __half;
    fused_add_rmsnorm_v1_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr()), reinterpret_cast<T*>(residual.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()), hidden, static_cast<float>(eps));
  } else {
    using T = __nv_bfloat16;
    fused_add_rmsnorm_v1_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr()), reinterpret_cast<T*>(residual.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()), hidden, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm (version 0 = scalar + smem tree, 1 = vectorized + warp shuffle)");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "in-place fused residual add + RMSNorm");
}
