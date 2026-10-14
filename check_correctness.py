"""Compare every implementation against an fp32 reference across shapes and dtypes.

Run:  python check_correctness.py
Writes results/correctness.csv and exits non-zero if anything fails.
"""
import csv
import os
import sys

import torch

import ops

EPS = 1e-6
TOKENS = [1, 7, 128, 1000, 4096]
HIDDEN = [64, 1024, 2048, 4096, 5120, 8192]  # 5120 = e.g. Llama-2-13B
ODD_HIDDEN = [100, 4100]  # not multiples of 8: only the scalar v0 kernel handles these
BLOCK_SIZES = [64, 128, 256, 512, 1024]
V1_BLOCK_SIZES = BLOCK_SIZES + [0]  # 0 = auto
TOL = {torch.float16: dict(rtol=1e-2, atol=1e-3), torch.bfloat16: dict(rtol=2e-2, atol=1e-2)}


def make_inputs(t, h, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, h, device="cuda", dtype=dtype, generator=g)
    r = torch.randn(t, h, device="cuda", dtype=dtype, generator=g)
    w = (torch.rand(h, device="cuda", generator=g) + 0.5).to(dtype)
    return x, r, w


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item() if a.numel() else 0.0


def main():
    os.makedirs("results", exist_ok=True)
    rows, failures = [], 0

    def record(op, impl, dtype, t, h, got, want, res_got=None, res_want=None):
        nonlocal failures
        err = max_err(got, want)
        ok = torch.allclose(got.float(), want.float(), **TOL[dtype])
        res_err = ""
        if res_got is not None:
            # residual is just a rounded add, so it should match (almost always bit for bit)
            res_err = f"{max_err(res_got, res_want):.3e}"
            ok = ok and torch.allclose(res_got.float(), res_want.float(), **TOL[dtype])
        failures += not ok
        rows.append(dict(op=op, impl=impl, dtype=str(dtype).replace("torch.", ""),
                         tokens=t, hidden=h, max_abs_err=f"{err:.3e}",
                         residual_max_abs_err=res_err, ok=ok))

    for dtype in (torch.float16, torch.bfloat16):
        for h in HIDDEN + ODD_HIDDEN:
            for t in TOKENS:
                x, r, w = make_inputs(t, h, dtype)
                want = ops.rmsnorm_ref(x, w, EPS)
                vec_ok = h % 8 == 0

                for bs in BLOCK_SIZES:
                    record("rmsnorm", f"mine v0 bs={bs}", dtype, t, h,
                           ops.make_rmsnorm_cuda(0, bs)(x, w, EPS), want)
                for bs in V1_BLOCK_SIZES if vec_ok else []:
                    record("rmsnorm", f"mine v1 bs={bs}", dtype, t, h,
                           ops.make_rmsnorm_cuda(1, bs)(x, w, EPS), want)
                if ops.HAVE_FLASHINFER and vec_ok:
                    record("rmsnorm", "flashinfer", dtype, t, h, ops.rmsnorm_flashinfer(x, w, EPS), want)

                if not vec_ok:
                    continue
                want_out, want_res = ops.fused_add_rmsnorm_ref(x, r, w, EPS)
                fused = {f"mine fused v1 bs={bs}": ops.make_fused_add_rmsnorm_cuda(bs) for bs in V1_BLOCK_SIZES}
                if ops.HAVE_FLASHINFER:
                    fused["flashinfer"] = ops.fused_add_rmsnorm_flashinfer
                for name, fn in fused.items():
                    xi, ri = x.clone(), r.clone()  # these kernels work in place
                    out, res = fn(xi, ri, w, EPS)
                    record("fused_add_rmsnorm", name, dtype, t, h, out, want_out, res, want_res)

    with open("results/correctness.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    print(f"{len(rows)} checks, {failures} failed")
    by_impl = {}
    for row in rows:
        key = (row["op"], row["impl"].split(" bs=")[0])
        prev = by_impl.get(key, (0.0, True))
        by_impl[key] = (max(prev[0], float(row["max_abs_err"])), prev[1] and row["ok"])
    for (op, impl), (err, ok) in sorted(by_impl.items()):
        print(f"  {op:18s} {impl:16s} worst max_abs_err={err:.2e}  {'ok' if ok else 'FAIL'}")
    for row in rows:
        if not row["ok"]:
            print("  FAIL:", row)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
