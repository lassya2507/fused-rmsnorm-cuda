# Fused Add + RMSNorm in CUDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lassya2507/fused-rmsnorm-cuda/blob/main/run_on_colab.ipynb)

I wanted to understand what actually makes a small LLM kernel fast, so I picked RMSNorm. It runs twice in every transformer layer, it's simple enough to write in an evening, and FlashInfer has a production version I can check myself against.

This repo has:

- **v0**: a straightforward CUDA RMSNorm (2-byte loads, shared-memory tree reduction)
- **v1**: the same algorithm with 128-bit vectorized loads and a warp-shuffle reduction
- **fused add + RMSNorm**: `residual += x; x = rmsnorm(residual) * w`, done in one kernel, in place, same semantics as `flashinfer.norm.fused_add_rmsnorm`
- a correctness check, a benchmark, and a small script that counts how many kernels each implementation launches

Everything below was measured on a free Colab **Tesla T4** (sm_75, 40 SMs, 4 MB L2), torch 2.11 / CUDA 13.0, FlashInfer 0.7.1, fp16. The full Colab logs are in `results/`.

## The one idea this whole thing rests on

RMSNorm does about 4 math ops per element and moves about 4 bytes per element (read 2, write 2 in fp16). A T4 can do far more math per byte than that, so the cores spend most of their time waiting on memory. The kernel is memory bound.

So "X is 1.4x faster than Y" isn't the number I care about. The number I care about is **effective bandwidth**: the minimum bytes the op has to move, divided by the time it took, compared to how fast this GPU can just copy memory. I measured a plain device-to-device copy at **243 GB/s** (the spec sheet says 320, nobody gets that in practice). If a norm kernel gets close to that, there's not much left to win.

Minimum bytes per element (fp16):

| op | reads | writes | bytes / element |
|---|---|---|---|
| RMSNorm | x | out | 4 |
| fused add + RMSNorm | x, residual | residual, out | 8 |

(plus the weight vector, which is tiny)

A few numbers show up slightly above 100%. That just means a copy isn't a perfect ceiling. I treat anything within 1–2% of each other as a tie.

## Results

### RMSNorm, large batch (16384 tokens)

| implementation | hidden 4096 | % of copy bw | hidden 8192 | % of copy bw |
|---|---|---|---|---|
| PyTorch eager (HF Llama style) | 10326 µs | 11% | 20562 µs | 11% |
| `F.rms_norm` | 1502 µs | 74% | 3246 µs | 68% |
| `torch.compile` | 1104 µs | 100% | 2283 µs | 97% |
| FlashInfer | 1098 µs | 101% | 2178 µs | 101% |
| mine v0 | 1563 µs | 71% | 3850 µs | 57% |
| **mine v1** | **1091 µs** | **101%** | **2185 µs** | **101%** |

v1 ends up level with FlashInfer and torch.compile. I'm not claiming I beat anything. For an op this simple there's a hard ceiling (memory bandwidth) and the good implementations all sit on it. What was interesting to me was how you get there, and how far off you are if you don't.

### Fused add + RMSNorm, large batch (16384 tokens)

| implementation | kernels launched | hidden 4096 | % of copy bw | hidden 8192 | % of copy bw |
|---|---|---|---|---|---|
| PyTorch eager | 9 | 11959 µs | 18% | 23831 µs | 18% |
| torch add, then my v1 norm (unfused) | 2 | 2735 µs | 81% | 5469 µs | 81% |
| `torch.compile` | 1 | 2282 µs | 97% | 4559 µs | 97% |
| FlashInfer | 1 | 2230 µs | 99% | 4463 µs | 99% |
| **mine fused v1** | **1** | **2264 µs** | **98%** | **4510 µs** | **98%** |

Kernel counts are from `count_kernels.py` (torch.profiler, 1024 x 4096).

## What I learned

### 1. Counting bytes predicted eager PyTorch almost exactly

The HF Llama RMSNorm does `x.float()`, `.pow(2)`, `.mean()`, `* rsqrt(...)`, `.to(half)`, `* w`. Each of those is its own kernel, and each one reads its whole input from DRAM and writes its whole output back. The profiler shows 8 kernels.

Adding up the bytes for those steps I got about 36 bytes per element, against a minimum of 4. So I expected eager to reach roughly 4/36 = **11%** of copy bandwidth. It measured **10–11%** at every large shape. For fused add + norm the same counting gives 8/42 = 19%, and it measured 18%.

That was the moment this clicked for me. The math in these kernels is basically free. What costs is every extra trip through DRAM.

### 2. Fusion is worth what the byte count says it's worth

If you do the residual add as its own kernel and then normalize, the residual gets written to DRAM and immediately read back. That's 10 bytes per element instead of 8, so I expected the unfused version to take about 1.25x as long.

Measured: **1.20–1.21x** across hidden 2048/4096/8192. Same kernel doing the norm in both cases, the only difference is the extra launch and the extra round trip. Fusing cut latency by about **17%**.

### 3. v0 → v1: wider loads and a cheaper reduction

v0 loads one fp16 value (2 bytes) per instruction and reduces with the classic shared-memory tree, which needs a `__syncthreads()` at every level.

v1 loads 8 values at once (one 16-byte `ld.global.v4`), and does the reduction with `__shfl_xor_sync` inside each warp, so there's only one small shared-memory step to combine the warps. I checked the generated PTX (`csrc/ptx_check.cu`) to make sure the compiler actually emitted 128-bit loads and not eight small ones. It did.

That took me from 71% to 101% at hidden 4096, and from 57% to 101% at hidden 8192 (1.4x and 1.8x faster).

### 4. Block size: one 16-byte vector per thread

I launch one thread block per token (row). The question is how many threads per block. I swept 64 to 1024 (`results/run1_fixed_bs256/colab_log.txt`), v1 RMSNorm, 16384 tokens:

| hidden | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| 2048 | 715 | 586 | 552 | **549** | 676 |
| 4096 | 1641 | 1451 | 1140 | 1090 | **1088** |
| 8192 | 3285 | 3271 | 2937 | 2234 | **2179** |

(µs, lower is better)

The winner every time is about hidden / 8 threads. Since each thread loads 8 values at a time, that means each thread does exactly one load per pass. Going bigger than that just leaves threads idle (hidden 2048 at 1024 threads, 3/4 of them have nothing to do and it gets slower). Going smaller means each thread loops several times.

My first full comparison accidentally used 256 threads everywhere and v1 only hit 75% at hidden 8192. After the sweep I made the launcher pick `min(1024, hidden / 8)` by default and that shape went to 101% (1.34x faster). Afterwards I looked at FlashInfer's `norm.cuh` and it uses the same rule: `block_size = min(1024, d / vec_size)`. Nice to see the measurement land on the same answer.

I don't fully know why 256 threads was so much worse at hidden 8192 specifically. My guess: with 256-thread blocks, 4 blocks fit on each SM instead of 1, so about 4x as many rows are in flight at once. The second pass re-reads x expecting it to still be in L2, and with that many 16 KB rows in flight it might not be. If x was fully re-read from DRAM I'd expect around 67%, and I saw 75%, which fits a partial miss. I haven't verified this with a profiler yet.

### 5. At decode batch sizes it's all launch overhead

With 1 token there's 1 thread block, so 39 of the T4's 40 SMs sit idle. Every decent implementation lands at about 5–6 µs, which is 1–3% of bandwidth. That's not a kernel problem, it's the fixed cost of launching a kernel.

| 1 token | hidden 4096 |
|---|---|
| PyTorch eager (8 launches) | 22.8 µs |
| `F.rms_norm` | 10.2 µs |
| mine v0 | 15.8 µs |
| torch.compile / FlashInfer / mine v1 | 5.1–5.5 µs |

Two things stood out here. Eager pays launch overhead 8 times. And v0 is 3–4.5x slower even at one token, because a single 256-thread block walking through a row 2 bytes at a time is just serial latency with nothing to hide it.

This is why serving engines care so much about fusion and CUDA graphs, and why FlashInfer's norm functions have an `enable_pdl` option (programmatic dependent launch, which lets the next kernel start launching before the previous one finishes).

### 6. Smaller things I noticed

- `F.rms_norm` (PyTorch's built-in) runs through a `vectorized_layer_norm_kernel` and topped out around 68–74% on this setup, noticeably behind `torch.compile` at large hidden sizes. I only looked at this on one GPU and one torch version.
- `torch.compile` generated a single fused Triton kernel (`triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0`) from the plain eager code and matched the hand-written kernels. Honestly impressive for zero effort.
- FlashInfer's fused kernel keeps the summed residual in shared memory for the second pass. Mine re-reads it from global memory and relies on L2. On the T4 the difference is about 1%, inside the noise, but it might matter more on a bigger GPU.

## Correctness

`check_correctness.py` compares every implementation against an fp32 reference: fp16 and bf16, hidden sizes 64 to 8192 (including 5120, and odd sizes like 100 and 4100 for v0), 1 to 4096 tokens, every block size. For the fused kernel it also checks the residual that gets written back.

1240 checks, 0 failures. Worst max abs error was 3.1e-2, the same for my kernels and for FlashInfer. That's one bf16 rounding step for outputs in the 4–8 range.

In the fused kernel the sum is rounded to fp16/bf16 before squaring, so the norm is computed on exactly the residual value the next layer will see.

## Running it

Easiest: click the Open in Colab badge at the top, pick a T4 runtime, and run all cells.

Locally it needs an NVIDIA GPU, PyTorch with CUDA, and `ninja`. FlashInfer is optional, everything skips it if it isn't installed.

```bash
pip install ninja flashinfer-python
python check_correctness.py      # first run JIT-compiles the extension, ~1 min
python bench.py                  # comparison, auto block size
python bench.py --sweep          # block-size sweep
python count_kernels.py          # kernels launched per implementation
```

How timing works: CUDA events around each call, median of 30–100 runs, and a 256 MB buffer gets zeroed before every call so the inputs aren't sitting in L2. These are cold-cache numbers on purpose.

## Files

```
csrc/rmsnorm_kernels.cuh   the kernels (v0, v1, fused) and the warp/block reductions
csrc/rmsnorm_binding.cu    PyTorch extension: checks, dtype dispatch, launch, auto block size
csrc/ptx_check.cu          compile this alone to inspect the PTX
ops.py                     every implementation behind one interface
check_correctness.py
bench.py
count_kernels.py
results/                   full logs from both Colab runs
```

## Things I'd do next

- Profile with Nsight Compute and look at actual DRAM bytes, mainly to test the L2 explanation for the 256-thread result above.
- Try keeping the row in shared memory or registers between the two passes like FlashInfer's fused kernel does, and see whether it matters on an A100/H100 where bandwidth is much higher.
- Run the same benchmark on a newer GPU. On a T4 everything good collapses onto the same bandwidth ceiling, and a faster memory system might separate them more.
- Look at decode more seriously: CUDA graphs, PDL, and fusing the norm into the neighbouring kernel instead of making the norm itself faster.
- Handle hidden sizes that aren't a multiple of 8 in the vectorized path (right now only v0 does).
