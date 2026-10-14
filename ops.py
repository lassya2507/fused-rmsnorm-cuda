"""All the RMSNorm implementations being compared, behind one small interface.

rmsnorm impls:            fn(x, w, eps) -> out
fused add + rmsnorm impls: fn(x, residual, w, eps) -> (out, new_residual)
                           (the in-place ones also overwrite x / residual, same as FlashInfer)
"""
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ext():
    """Compile (first call only, ~1 min) and return the CUDA extension."""
    global _ext
    if _ext is None:
        _ext = load(
            name="rmsnorm_cuda",
            sources=[os.path.join(_HERE, "csrc", "rmsnorm_binding.cu")],
            extra_include_paths=[os.path.join(_HERE, "csrc")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------
# references (fp32 math, single rounding at the end). Used for correctness.
# ---------------------------------------------------------------------------
def rmsnorm_ref(x, w, eps):
    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv_rms * w.float()).to(x.dtype)


def fused_add_rmsnorm_ref(x, residual, w, eps):
    new_residual = (residual.float() + x.float()).to(x.dtype)
    return rmsnorm_ref(new_residual, w, eps), new_residual


# ---------------------------------------------------------------------------
# PyTorch baselines
# ---------------------------------------------------------------------------
def rmsnorm_eager(x, w, eps):
    # the way HF Llama writes it: several separate elementwise/reduction ops
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return w * xf.to(x.dtype)


def fused_add_rmsnorm_eager(x, residual, w, eps):
    residual = residual + x
    return rmsnorm_eager(residual, w, eps), residual


def rmsnorm_F(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


_compiled = {}


def _compiled_fn(name, fn):
    if name not in _compiled:
        # one compile per shape (dynamic=False); raise the recompile limit so
        # dynamo doesn't silently fall back to eager after 8 shapes
        for attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, attr):
                setattr(torch._dynamo.config, attr, 512)
        _compiled[name] = torch.compile(fn, dynamic=False)
    return _compiled[name]


def rmsnorm_compiled(x, w, eps):
    return _compiled_fn("rmsnorm", rmsnorm_eager)(x, w, eps)


def fused_add_rmsnorm_compiled(x, residual, w, eps):
    return _compiled_fn("fused", fused_add_rmsnorm_eager)(x, residual, w, eps)


# ---------------------------------------------------------------------------
# FlashInfer (optional)
# ---------------------------------------------------------------------------
try:
    import flashinfer  # noqa: F401
    from flashinfer.norm import fused_add_rmsnorm as _fi_fused
    from flashinfer.norm import rmsnorm as _fi_rmsnorm

    HAVE_FLASHINFER = True
except Exception:  # not installed / unsupported env
    HAVE_FLASHINFER = False


def rmsnorm_flashinfer(x, w, eps):
    return _fi_rmsnorm(x, w, eps)


def fused_add_rmsnorm_flashinfer(x, residual, w, eps):
    _fi_fused(x, residual, w, eps)  # in place
    return x, residual


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------
def make_rmsnorm_cuda(version, block_size):
    def fn(x, w, eps):
        return ext().rmsnorm(x, w, eps, version, block_size)

    return fn


def make_fused_add_rmsnorm_cuda(block_size):
    def fn(x, residual, w, eps):
        ext().fused_add_rmsnorm(x, residual, w, eps, block_size)  # in place
        return x, residual

    return fn


def _bs_label(block_size):
    return "auto bs" if block_size <= 0 else f"bs={block_size}"


def make_unfused_add_then_rmsnorm(block_size):
    # same math as the fused kernel, but as two launches: a torch add, then my v1 norm.
    # the residual makes an extra round trip through DRAM, which is what fusion removes.
    def fn(x, residual, w, eps):
        residual = residual + x
        return ext().rmsnorm(residual, w, eps, 1, block_size), residual

    return fn


def has_F_rms_norm():
    return hasattr(F, "rms_norm")


def rmsnorm_impls(block_size=0, v0_block_size=256):
    impls = {
        "torch eager": rmsnorm_eager,
        "torch compile": rmsnorm_compiled,
    }
    if has_F_rms_norm():
        impls["F.rms_norm"] = rmsnorm_F
    if HAVE_FLASHINFER:
        impls["flashinfer"] = rmsnorm_flashinfer
    impls[f"mine v0 (bs={v0_block_size})"] = make_rmsnorm_cuda(0, v0_block_size)
    impls[f"mine v1 ({_bs_label(block_size)})"] = make_rmsnorm_cuda(1, block_size)
    return impls


def fused_impls(block_size=0):
    impls = {
        "torch eager": fused_add_rmsnorm_eager,
        "torch compile": fused_add_rmsnorm_compiled,
    }
    if HAVE_FLASHINFER:
        impls["flashinfer"] = fused_add_rmsnorm_flashinfer
    impls[f"unfused: add + mine v1 ({_bs_label(block_size)})"] = make_unfused_add_then_rmsnorm(block_size)
    impls[f"mine fused v1 ({_bs_label(block_size)})"] = make_fused_add_rmsnorm_cuda(block_size)
    return impls
