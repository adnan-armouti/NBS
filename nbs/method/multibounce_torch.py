"""Differentiable (torch) multibounce WiFi CSI renderer.

Path GEOMETRY (specular points, lengths, incidence angles) is precomputed once in numpy
(image method == mmIR renderer/specular/image_method.py). The render is then differentiable
in the per-plane MATERIAL (Fresnel reflectance), which is what we recover/transfer:

    H[link, f] = sum_paths ( prod_bounces Gamma(eps[plane], cos_i) ) / L * exp(-j 2pi f L / c)

Gradients flow to eps (per plane). The geometry is fixed (known scaffold), so it stays numpy.
"""
import math
import numpy as np
import torch
from method import multibounce as MB

C_LIGHT = 299792458.0


def precompute_paths(tx, rx_list, planes, order=2):
    """Per-link list of (length, [(plane_idx, cos_i), ...]) — the differentiable-render input."""
    return [MB.link_paths_geom(np.asarray(tx, float), np.asarray(r, float), planes, order)
            for r in rx_list]


def fresnel_te(eps_r, cos_i):
    """TE Fresnel reflection coefficient (real eps), differentiable in eps_r."""
    cos_i = cos_i.clamp(1e-3, 1.0)
    s = torch.sqrt((eps_r - (1.0 - cos_i ** 2)).clamp(min=1e-9))
    return (cos_i - s) / (cos_i + s + 1e-12)


def flatten_paths(paths_per_link, device="cpu"):
    """Flatten the per-link path lists into batched tensors for the vectorized/CUDA render.
    Returns dict of (n_paths,) tensors: link, L, nb (#bounces), p0,c0,p1,c1 (padded)."""
    link, L, nb, p0, c0, p1, c1 = [], [], [], [], [], [], []
    for i, paths in enumerate(paths_per_link):
        for Lp, bounces in paths:
            link.append(i); L.append(Lp); nb.append(len(bounces))
            p0.append(bounces[0][0] if len(bounces) >= 1 else 0)
            c0.append(bounces[0][1] if len(bounces) >= 1 else 1.0)
            p1.append(bounces[1][0] if len(bounces) >= 2 else 0)
            c1.append(bounces[1][1] if len(bounces) >= 2 else 1.0)
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return {"link": t(link, torch.long), "L": t(L, torch.float64), "nb": t(nb, torch.long),
            "p0": t(p0, torch.long), "c0": t(c0, torch.float64),
            "p1": t(p1, torch.long), "c1": t(c1, torch.float64), "n_links": len(paths_per_link)}


def render_csi_vec(flat, eps, sc_hz, device="cpu"):
    """Vectorized, differentiable CSI render (no Python loops over paths/links). GPU-ready."""
    f = torch.as_tensor(sc_hz, dtype=torch.float64, device=device)
    g0 = torch.where(flat["nb"] >= 1, fresnel_te(eps[flat["p0"]], flat["c0"]), torch.ones_like(flat["c0"]))
    g1 = torch.where(flat["nb"] >= 2, fresnel_te(eps[flat["p1"]], flat["c1"]), torch.ones_like(flat["c1"]))
    gamma = g0 * g1                                                   # (n_paths,)
    phase = torch.exp(-1j * 2 * math.pi * flat["L"][:, None] * (f[None, :] / C_LIGHT))
    contrib = (gamma / flat["L"]).to(torch.complex128)[:, None] * phase   # (n_paths, n_sc)
    H = torch.zeros(flat["n_links"], len(sc_hz), dtype=torch.complex128, device=device)
    H.index_add_(0, flat["link"], contrib)
    return H


def render_csi(paths_per_link, eps, sc_hz, device="cpu"):
    """Complex CSI (n_links, n_sc), differentiable in eps (per-plane permittivity, tensor)."""
    f = torch.as_tensor(sc_hz, dtype=torch.float64, device=device)
    H = torch.zeros(len(paths_per_link), len(sc_hz), dtype=torch.complex128, device=device)
    for i, paths in enumerate(paths_per_link):
        acc = torch.zeros(len(sc_hz), dtype=torch.complex128, device=device)
        for L, bounces in paths:
            gamma = torch.ones((), dtype=torch.float64, device=device)
            for pid, cos_i in bounces:
                gamma = gamma * fresnel_te(eps[pid], torch.tensor(cos_i, dtype=torch.float64, device=device))
            phase = torch.exp(-1j * 2 * math.pi * f * (L / C_LIGHT))
            acc = acc + (gamma.to(torch.complex128) / L) * phase
        H[i] = acc
    return H
