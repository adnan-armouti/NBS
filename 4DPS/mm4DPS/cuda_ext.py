"""Both CUDA extensions: how they are built, how they are loaded, and their
autograd wrappers.

Merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C6 from four modules --
`cuda/setup.py`, `cuda/__init__.py`, `splat/cuda/setup.py` and
`splat/cuda/__init__.py`. The two setup scripts shared only seventeen lines
verbatim but were structurally identical: same CUDA-version waiver, same
`setup()` shape, differing in name, sources and flags. They are one table now.

    hupr_cuda    our fused kernels               mm4DPS/cuda/
    mm3DPS_cuda  upstream's BSDF + scatter splat  mm4DPS/splat/cuda/

BOTH are required. `mm3DPS_cuda` backs `splat_scatter`, and a tree with only
`hupr_cuda` passes every check in the repo and then dies four stages into a
chain.

NO `.so` MOVED, and none needs to. Both extensions are loaded by explicit file
path out of their own source directory, which is where `build_ext --inplace`
drops them -- the technique `splat/cuda/__init__.py` already used. That is what
lets those two directories stop being Python packages while the compiled objects
stay exactly where the build puts them. `cuda/__init__.py` existed only so that
`from mm4DPS.cuda import hupr_cuda` would resolve; loading by path removes the
need for it.

THE BUILD STAMP NOW HAS A READER. `cuda/setup.py` recorded a SHA-256 of each
source after a successful build so that "was this .so built from these exact
sources" could be answered by content rather than by mtime -- mtime being
useless here, since `git checkout` rewrites it on every file whose content
changes. The stamp was written and **nothing ever read it**, and only one of the
two extensions wrote one. Both write one now and
`python -m mm4DPS.qa check-environment` verifies them, because a stale
extension does not present as "you need to rebuild": it presents as
`hupr_static_gain_splat_backward(): incompatible function arguments`, a pybind
signature error that names a function and says nothing about compilation. That
has cost this project twice in two days.

Build both:  make build-cuda        (or `python -m mm4DPS.cuda_ext build`)
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

# torch must be imported before the extension .so so that libc10/libtorch are
# preloaded into the process (otherwise ImportError on libc10.so).
import torch  # noqa: F401

__all__ = ["ext", "is_available", "scatter_splat", "splat_scatter",
           "step5_fused", "bsdf_step4_forward", "load", "load_hupr_cuda",
           "build", "stamp_status", "EXTENSIONS"]

_PACKAGE = Path(__file__).resolve().parent
STAMP = ".build_stamp.json"

#: Each extension: where its sources live, what they are, and the flags they
#: need. `--use_fast_math` is deliberately absent from BOTH, for the same
#: measured reason: near-specular GGX (alpha ~ 4e-3) amplifies 1e-7-level
#: fast-math error ~1e5x, and the compiled step-4 kernel deviated from its own
#: source math by up to 4.3x on 72% of paths (measured 2026-08-17). The step-5
#: carrier sincos also lost range reduction at ~1e4 rad phases. IEEE fp32
#: restores agreement with the PyTorch reference.
EXTENSIONS = {
    "hupr_cuda": {
        "directory": _PACKAGE / "cuda",
        "sources": ("bindings.cpp", "kernels.cu"),
        "include": lambda d: [str(d)],
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17", "--extended-lambda",
                 "-U__CUDA_NO_HALF_OPERATORS__"],
    },
    "mm3DPS_cuda": {
        "directory": _PACKAGE / "splat" / "cuda",
        "sources": ("bindings.cpp", "bsdf_forward.cu", "bsdf_backward.cu",
                    "scatter_splat.cu"),
        # mm4DPS/cuda holds sincos_reduced.cuh, the one definition of a symbol
        # ours and used by this tree's scatter_splat.cu.
        "include": lambda d: [str(d), str(d.parent.parent / "cuda")],
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17", "--extended-lambda", "--ptxas-options=-v",
                 "--expt-relaxed-constexpr", "-lineinfo"],
    },
}


# ----------------------------------------------------------------------
# Loading, by explicit file path
# ----------------------------------------------------------------------
def load(name: str):
    """Import a built extension from its own source directory.

    `build_ext --inplace` drops the .so next to the sources. Loading it by path
    rather than as a package submodule is what lets those directories hold no
    Python at all. Falls back to sys.path so an installed wheel still works.
    """
    directory = EXTENSIONS[name]["directory"]
    if directory.is_dir():
        for entry in sorted(os.listdir(directory)):
            if entry.startswith(name) and entry.endswith(".so"):
                spec = importlib.util.spec_from_file_location(
                    name, str(directory / entry))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[name] = module
                return module
    return importlib.import_module(name)


ext = None
_LOAD_ERROR = None
try:
    ext = load("mm3DPS_cuda")
except Exception as exc:  # ImportError, OSError, FileNotFoundError, etc.
    ext = None
    _LOAD_ERROR = repr(exc)


_HUPR = None
_HUPR_ERROR = None


def load_hupr_cuda():
    """`hupr_cuda`, or a RuntimeError naming the build command.

    Raised at CALL time, not import time, and it names the command: a missing
    extension must not turn into a silently smaller test suite, which is exactly
    how this went wrong once already (ledger R022).
    """
    global _HUPR, _HUPR_ERROR
    if _HUPR is None and _HUPR_ERROR is None:
        try:
            _HUPR = load("hupr_cuda")
        except Exception as exc:
            _HUPR_ERROR = exc
    if _HUPR is None:
        raise RuntimeError(
            f"hupr_cuda extension unavailable: {_HUPR_ERROR}. Build it with "
            f"`make build-cuda` (or TORCH_CUDA_ARCH_LIST=8.9 python -m "
            f"mm4DPS.cuda_ext build).")
    return _HUPR


# ----------------------------------------------------------------------
# Building, and the stamp that says what was compiled
# ----------------------------------------------------------------------
def _source_hashes(name: str) -> dict[str, str]:
    """Content hashes of the sources AND every header they can include.

    A header edit (a `.cuh` in the extension's directory, or in an include
    directory it names) changes the compiled code as surely as a `.cu` edit;
    hashing only the listed sources let a stale .so present as current.
    """
    spec = EXTENSIONS[name]
    hashes = {source: hashlib.sha256((spec["directory"] / source).read_bytes()).hexdigest()
              for source in spec["sources"]}
    for include in spec["include"](spec["directory"]):
        for header in sorted(Path(include).glob("*.cuh")):
            hashes[str(header.relative_to(_PACKAGE))] = hashlib.sha256(header.read_bytes()).hexdigest()
    return hashes


def stamp_status() -> dict[str, str]:
    """Per extension: built and current, built but STALE, or not built.

    STALE is the case worth having: it means the .so exists and was compiled
    from different source content than is on disk now. mtime cannot tell you
    this -- `git checkout` rewrites mtimes on content change, so a branch switch
    makes a good build look stale and restoring an older source leaves mtimes
    that look fine.
    """
    out = {}
    for name, spec in EXTENSIONS.items():
        directory = spec["directory"]
        built = [e for e in sorted(os.listdir(directory))
                 if e.startswith(name) and e.endswith(".so")] if directory.is_dir() else []
        if not built:
            out[name] = "NOT BUILT -- run `make build-cuda`"
            continue
        stamp_path = directory / STAMP
        if not stamp_path.is_file():
            out[name] = f"built ({built[0]}), no stamp -- provenance unknown"
            continue
        recorded = json.loads(stamp_path.read_text()).get("sources", {})
        current = _source_hashes(name)
        changed = [s for s in current if recorded.get(s) != current[s]]
        out[name] = (f"built and current ({built[0]})" if not changed
                     else f"STALE: {', '.join(changed)} changed since the build "
                          f"-- run `make build-cuda`")
    return out


def build(names=()) -> int:
    """Compile the named extensions in place, then stamp what went in.

    The stamp is written only after `setup()` returns without raising, i.e. only
    on a successful build.
    """
    import torch.utils.cpp_extension as cpp_ext

    # PyTorch 2.7.1+cu118 was compiled against CUDA 11.8 and the system has
    # 12.8. The objects run fine (cu118 libs are forward compatible with the
    # 12.x driver, and nothing here uses a cu12-only feature), so skip the
    # pre-build check. Both extensions have always made this same waiver.
    cpp_ext._check_cuda_version = lambda *a, **kw: None
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    for name in (names or tuple(EXTENSIONS)):
        spec = EXTENSIONS[name]
        directory = spec["directory"]
        extension = CUDAExtension(
            name=name,
            sources=[str(directory / s) for s in spec["sources"]],
            include_dirs=spec["include"](directory),
            extra_compile_args={"cxx": spec["cxx"], "nvcc": spec["nvcc"]},
        )
        cwd = os.getcwd()
        os.chdir(directory)
        try:
            setup(name=name, ext_modules=[extension],
                  cmdclass={"build_ext": BuildExtension},
                  script_args=["build_ext", "--inplace"])
        finally:
            os.chdir(cwd)
        (directory / STAMP).write_text(
            json.dumps({"sources": _source_hashes(name)}, indent=2) + "\n")
        print(f"built {name} and stamped {len(_source_hashes(name))} sources and headers")
    return 0

# --------------------------------------------------------------------
# The mm3DPS wrappers: autograd around the fused kernels
# --------------------------------------------------------------------

def is_available() -> bool:
    """True if the compiled CUDA extension is importable."""
    return ext is not None




def scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag):
    """In-place atomic scatter splat. Equivalent to:

        rp_real.view(-1).scatter_add_(0, flat_idx, contrib_real)
        rp_imag.view(-1).scatter_add_(0, flat_idx, contrib_imag)

    but issues a single fused CUDA kernel launch.
    """
    if ext is None:
        raise RuntimeError(
            "mm3DPS_cuda extension is not built. "
            "Run: cd mm4DPS/splat/cuda && python setup.py build_ext --inplace"
        )
    ext.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)


class ScatterSplatFn(torch.autograd.Function):
    """Phase D autograd wrapper for the scatter_splat CUDA kernel.

    Forward: writes contrib_real / contrib_imag into rp_real / rp_imag
    via atomicAdd, returning the accumulated rp tensors.

    Backward: grad flows to contrib_real / contrib_imag via a gather
    (`grad_contrib[i] = grad_rp.view(-1)[flat_idx[i]]`) — this is
    PyTorch's standard scatter_add backward, just faster because we do
    a single fused gather per output instead of two.
    """

    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, flat_idx,
                out_shape_tx, out_shape_rx, out_shape_k):
        contrib_real = contrib_real.contiguous()
        contrib_imag = contrib_imag.contiguous()
        flat_idx     = flat_idx.contiguous()
        rp_real = torch.zeros(out_shape_tx, out_shape_rx, out_shape_k,
                              device=contrib_real.device,
                              dtype=contrib_real.dtype)
        rp_imag = torch.zeros_like(rp_real)
        ext.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)
        ctx.save_for_backward(flat_idx)
        ctx.rp_numel = rp_real.numel()
        return rp_real, rp_imag

    @staticmethod
    def backward(ctx, grad_rp_real, grad_rp_imag):
        (flat_idx,) = ctx.saved_tensors
        # Gather: grad_contrib[i] = grad_rp.view(-1)[flat_idx[i]]
        grad_real_flat = grad_rp_real.reshape(-1)
        grad_imag_flat = grad_rp_imag.reshape(-1)
        grad_contrib_real = grad_real_flat[flat_idx]
        grad_contrib_imag = grad_imag_flat[flat_idx]
        return grad_contrib_real, grad_contrib_imag, None, None, None, None


def splat_scatter(contrib_real, contrib_imag, flat_idx, out_shape):
    """Autograd-friendly entry point. Returns (rp_real, rp_imag)."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return ScatterSplatFn.apply(
        contrib_real, contrib_imag, flat_idx,
        out_shape[0], out_shape[1], out_shape[2],
    )


class Step5FusedFn(torch.autograd.Function):
    """Phase D fused Step-5 autograd.Function.

    Forward: takes (w_full, phi_carrier, n_peak, psf_real, psf_imag, K,
    w_threshold), returns (rp_real, rp_imag). The kernel fuses:
      - carrier phasor computation (sin/cos of phi)
      - PSF table lookup with linear interpolation
      - bin index computation (floor + offset mod K)
      - contrib = carrier × psf complex multiply
      - atomic scatter into rp_real, rp_imag
    All intermediates stay in registers. No (SPREAD, P) contrib
    materialization.

    Backward: only computes grad_w_full. phi_carrier, n_peak, and the
    PSF tables are treated as non-differentiable inputs (which matches
    the production path where detach_phase=True and n_peak is
    explicitly detached).
    """

    @staticmethod
    def forward(ctx, w_full, phi_carrier, n_peak, psf_real, psf_imag,
                K, w_threshold):
        w_full      = w_full.contiguous()
        phi_carrier = phi_carrier.contiguous()
        n_peak      = n_peak.contiguous()
        psf_real    = psf_real.contiguous()
        psf_imag    = psf_imag.contiguous()
        rp_real, rp_imag = ext.step5_fused_forward(
            w_full, phi_carrier, n_peak, psf_real, psf_imag,
            K, float(w_threshold),
        )
        ctx.save_for_backward(w_full, phi_carrier, n_peak, psf_real, psf_imag)
        ctx.w_threshold = float(w_threshold)
        return rp_real, rp_imag

    @staticmethod
    def backward(ctx, grad_rp_real, grad_rp_imag):
        w_full, phi_carrier, n_peak, psf_real, psf_imag = ctx.saved_tensors
        grad_w = ext.step5_fused_backward(
            grad_rp_real.contiguous(), grad_rp_imag.contiguous(),
            w_full, phi_carrier, n_peak, psf_real, psf_imag,
            ctx.w_threshold,
        )
        # w_full, phi_carrier, n_peak, psf_real, psf_imag, K, w_threshold
        return grad_w, None, None, None, None, None, None


def step5_fused(w_full, phi_carrier, n_peak, psf_real, psf_imag,
                K, w_threshold=1e-20):
    """Autograd-friendly entry point for the fused Step-5 kernel."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return Step5FusedFn.apply(
        w_full, phi_carrier, n_peak, psf_real, psf_imag, K, w_threshold,
    )


class BSDFStep4ForwardFn(torch.autograd.Function):
    """Phase B: fused Step-4 BSDF forward wrapped as an autograd.Function.

    Forward: calls the CUDA kernel. Backward: falls back to the PyTorch
    reference implementation. Phase C will replace the backward with an
    analytical CUDA kernel.

    This is used only when `use_cuda_kernels=True` is passed through
    render_factorized; the pure-PyTorch path remains fully supported.
    """

    @staticmethod
    def forward(ctx,
                wi, wi_r, wo, n_eff, s_in,
                cos_i, cos_o, lambda_i, lambda_o,
                alpha_sq, kappa_SPM, norm_SPM, eps_factor,
                eps_real_m, eps_imag_m, thickness_m,
                E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
                tau_eff, k_wave):
        assert ext is not None, "mm3DPS_cuda extension not loaded"
        # Ensure contiguous float32 — cheap if already.
        args = [t.contiguous() for t in [
            wi, wi_r, wo, n_eff, s_in,
            cos_i, cos_o, lambda_i, lambda_o,
            alpha_sq, kappa_SPM, norm_SPM, eps_factor,
            eps_real_m, eps_imag_m, thickness_m,
            E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
            tau_eff,
        ]]
        ctx.k_wave = float(k_wave)
        f_cos = ext.bsdf_step4_forward(*args, float(k_wave))
        # Save for backward. Phase B uses the PyTorch reference for backward
        # (so we need the inputs to rerun Step 4 in autograd mode).
        ctx.save_for_backward(*args)
        return f_cos

    @staticmethod
    def backward(ctx, grad_f_cos):
        # Phase C: analytical CUDA backward via bsdf_step4_backward.
        saved = ctx.saved_tensors
        grad_f_cos = grad_f_cos.contiguous()
        grads = ext.bsdf_step4_backward(
            grad_f_cos,
            saved[0],  saved[1],  saved[2],  saved[3],  saved[4],
            saved[5],  saved[6],  saved[7],  saved[8],
            saved[9],  saved[10], saved[11], saved[12],
            saved[13], saved[14], saved[15],
            saved[16], saved[17], saved[18], saved[19],
            saved[20],
            ctx.k_wave,
        )
        return tuple(grads) + (None,)


def bsdf_step4_forward(
    wi, wi_r, wo, n_eff, s_in,
    cos_i, cos_o, lambda_i, lambda_o,
    alpha_sq, kappa_SPM, norm_SPM, eps_factor,
    eps_real_m, eps_imag_m, thickness_m,
    E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
    tau_eff, k_wave,
):
    """Autograd-friendly entry point for the CUDA Step-4 BSDF kernel."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return BSDFStep4ForwardFn.apply(
        wi, wi_r, wo, n_eff, s_in,
        cos_i, cos_o, lambda_i, lambda_o,
        alpha_sq, kappa_SPM, norm_SPM, eps_factor,
        eps_real_m, eps_imag_m, thickness_m,
        E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
        tau_eff, k_wave,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="mm4DPS.cuda_ext", description=__doc__)
    parser.add_argument("command", choices=("build", "status"))
    parser.add_argument("--only", nargs="*", default=(), choices=tuple(EXTENSIONS),
                        help="build just these (default: both, and both are required)")
    args = parser.parse_args()
    if args.command == "status":
        for name, status in stamp_status().items():
            print(f"{name:14s} {status}")
        return 0
    return build(tuple(args.only))


if __name__ == "__main__":
    raise SystemExit(main())
