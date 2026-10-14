"""How many GPU kernels does each implementation actually launch?

Run:  python count_kernels.py
Uses torch.profiler on one call per implementation (after warmup).
"""
import torch
from torch.profiler import ProfilerActivity, profile

import ops

EPS = 1e-6
T, H = 1024, 4096


def kernels_launched(fn, setup=lambda: ()):
    for _ in range(3):  # warmup (and torch.compile compilation)
        fn(*setup())
    args = setup()  # fresh inputs prepared outside the profiled region
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return names


def main():
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)
    r = torch.randn_like(x)
    w = (torch.rand(H, device="cuda") + 0.5).half()

    print(f"shape {T} x {H}, fp16\n")
    for name, fn in ops.rmsnorm_impls().items():
        names = kernels_launched(lambda: fn(x, w, EPS))
        print(f"rmsnorm  | {name:30s} | {len(names)} kernel(s)")
        for n in names:
            print(f"         |   {n[:110]}")
    for name, fn in ops.fused_impls().items():
        names = kernels_launched(lambda xi, ri: fn(xi, ri, w, EPS), setup=lambda: (x.clone(), r.clone()))
        print(f"fused    | {name:30s} | {len(names)} kernel(s)")
        for n in names:
            print(f"         |   {n[:110]}")


if __name__ == "__main__":
    main()
