"""Latency + effective memory bandwidth for every implementation.

Usage:
  python bench.py --sweep              # block-size sweep for my v1 / fused kernels
  python bench.py                      # compare against torch / torch.compile / FlashInfer
                                       # (my kernels use the auto block size; --block-size N to force one)
  python bench.py --no-compile         # skip torch.compile (it recompiles per shape, takes a few min)

How timing works:
  - each call is timed with a pair of CUDA events, median over many calls
  - before every call a 256 MB buffer is zeroed to push the inputs out of L2,
    so numbers are "cold cache" (the honest case for big activations)
  - effective bandwidth = minimum bytes the op has to move / time
      rmsnorm:           read x + read w + write out
      fused add+rmsnorm: read x + read residual + write residual + write out + read w
  - % is against a measured device-to-device copy, which is roughly the best
    bandwidth you can actually get on this GPU
"""
import argparse
import csv
import os
import statistics

import torch

import ops

EPS = 1e-6
TOKENS = [1, 16, 128, 1024, 4096, 16384]
HIDDEN = [2048, 4096, 8192]
SWEEP_BLOCK_SIZES = [64, 128, 256, 512, 1024]

# rough spec-sheet DRAM bandwidth in GB/s, only printed for context
SPEC_BW = {
    "T4": 320, "L4": 300, "A10G": 600, "A10": 600, "L40S": 864,
    "A100-SXM4-40GB": 1555, "A100-PCIE-40GB": 1555, "A100-SXM4-80GB": 2039, "A100 80GB PCIe": 1935,
    "H100 80GB HBM3": 3350, "H100 PCIe": 2000, "RTX 4090": 1008, "RTX 3090": 936,
}

_flush = None
_start = _end = None


def timed(fn, setup=None, iters=100, warmup=10):
    """Median time of fn() in microseconds."""
    global _flush, _start, _end
    if _flush is None:
        _flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
        _start = torch.cuda.Event(enable_timing=True)
        _end = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(warmup + iters):
        if setup is not None:
            setup()
        _flush.zero_()
        _start.record()
        fn()
        _end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(_start.elapsed_time(_end) * 1000.0)  # ms -> us
    return statistics.median(times)


def copy_bandwidth_gbs():
    n = 512 * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    us = timed(lambda: dst.copy_(src), iters=20, warmup=3)
    return 2 * n / (us * 1e-6) / 1e9


def env_info(copy_bw):
    p = torch.cuda.get_device_properties(0)
    spec = next((v for k, v in SPEC_BW.items() if k in p.name), None)
    lines = [
        f"gpu: {p.name} (sm_{p.major}{p.minor}, {p.multi_processor_count} SMs, "
        f"L2 {getattr(p, 'L2_cache_size', 0) / 2**20:.0f} MB)",
        f"torch: {torch.__version__}, cuda: {torch.version.cuda}",
        f"flashinfer: {getattr(__import__('flashinfer'), '__version__', '?') if ops.HAVE_FLASHINFER else 'not installed'}",
        f"measured copy bandwidth: {copy_bw:.0f} GB/s" + (f"  (spec sheet ~{spec} GB/s)" if spec else ""),
    ]
    return "\n".join(lines)


def make_inputs(t, h, dtype):
    x = torch.randn(t, h, device="cuda", dtype=dtype)
    r = torch.randn(t, h, device="cuda", dtype=dtype)
    w = (torch.rand(h, device="cuda") + 0.5).to(dtype)
    return x, r, w


def bytes_moved(op, t, h, elt):
    if op == "rmsnorm":
        return (2 * t * h + h) * elt
    return (4 * t * h + h) * elt


def run_rmsnorm(fn, x, w, iters):
    return timed(lambda: fn(x, w, EPS), iters=iters)


def run_fused(fn, x0, r0, w, iters):
    x, r = x0.clone(), r0.clone()

    def setup():  # in-place kernels: restore inputs before every call (outside the timed region)
        x.copy_(x0)
        r.copy_(r0)

    return timed(lambda: fn(x, r, w, EPS), setup=setup, iters=iters)


def iters_for(t, h):
    return 30 if t * h >= 4096 * 8192 else 100


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)


def print_table(title, rows, impls, value):
    print(f"\n### {title}\n")
    print("| tokens x hidden | " + " | ".join(impls) + " |")
    print("|---|" + "---|" * len(impls))
    shapes = sorted({(r["tokens"], r["hidden"]) for r in rows}, key=lambda s: (s[1], s[0]))
    for t, h in shapes:
        cells = []
        for name in impls:
            match = [r for r in rows if r["tokens"] == t and r["hidden"] == h and r["impl"] == name]
            cells.append(value(match[0]) if match else "-")
        print(f"| {t} x {h} | " + " | ".join(cells) + " |")


def sweep(dtype):
    rows = []
    for h in HIDDEN:
        for t in TOKENS:
            x, r, w = make_inputs(t, h, dtype)
            for bs in SWEEP_BLOCK_SIZES:
                us = run_rmsnorm(ops.make_rmsnorm_cuda(1, bs), x, w, iters_for(t, h))
                rows.append(dict(op="rmsnorm", impl=f"bs={bs}", tokens=t, hidden=h, us=round(us, 2)))
                us = run_fused(ops.make_fused_add_rmsnorm_cuda(bs), x, r, w, iters_for(t, h))
                rows.append(dict(op="fused", impl=f"bs={bs}", tokens=t, hidden=h, us=round(us, 2)))
            print(f"  swept {t} x {h}")
    write_csv("results/sweep.csv", rows)
    names = [f"bs={bs}" for bs in SWEEP_BLOCK_SIZES]
    for op in ("rmsnorm", "fused"):
        sub = [r for r in rows if r["op"] == op]
        print_table(f"block-size sweep, my {op} v1 kernel, latency in us (lower is better)", sub, names,
                    lambda r: f"{r['us']:.1f}")


def compare(dtype, block_size, use_compile, copy_bw):
    elt = torch.finfo(dtype).bits // 8
    results = {}
    for op, impls_fn, runner in (("rmsnorm", ops.rmsnorm_impls, run_rmsnorm),
                                 ("fused", ops.fused_impls, run_fused)):
        impls = impls_fn(block_size)
        if not use_compile:
            impls.pop("torch compile", None)
        rows = []
        for h in HIDDEN:
            for t in TOKENS:
                x, r, w = make_inputs(t, h, dtype)
                for name, fn in impls.items():
                    args = (fn, x, w) if op == "rmsnorm" else (fn, x, r, w)
                    us = runner(*args, iters_for(t, h))
                    gbs = bytes_moved(op, t, h, elt) / (us * 1e-6) / 1e9
                    rows.append(dict(op=op, impl=name, tokens=t, hidden=h, us=round(us, 2),
                                     gbs=round(gbs, 1), pct_of_copy=round(100 * gbs / copy_bw, 1)))
                print(f"  {op}: {t} x {h} done")
        write_csv(f"results/bench_{op}.csv", rows)
        results[op] = (rows, list(impls))
    for op, (rows, names) in results.items():
        print_table(f"{op}: latency in us", rows, names, lambda r: f"{r['us']:.1f}")
        print_table(f"{op}: effective bandwidth, % of measured copy bandwidth", rows, names,
                    lambda r: f"{r['pct_of_copy']:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--block-size", type=int, default=0, help="0 = auto (one 16-byte vector per thread)")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ops.ext()  # compile up front so it isn't inside a measurement
    copy_bw = copy_bandwidth_gbs()
    info = env_info(copy_bw)
    print(info)
    with open("results/env.txt", "w") as f:
        f.write(info + f"\ndtype: {args.dtype}\n")

    if args.sweep:
        sweep(dtype)
    else:
        compare(dtype, args.block_size, not args.no_compile, copy_bw)


if __name__ == "__main__":
    main()
