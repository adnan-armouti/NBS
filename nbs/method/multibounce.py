"""Lightweight multibounce via the IMAGE METHOD, to render WiFi multipath that the
single-bounce 3DPS renderer cannot. Fast (deterministic, ~tens of paths per link),
differentiable in the per-plane material, and the classic indoor-channel model.

Pipeline: fit the room's dominant planes (RANSAC on the scaffold) -> for each Tx-Rx link,
enumerate LOS + 1st + 2nd-order specular reflections via source images -> sum the paths
H(f) = sum_p Gamma_p / L_p * exp(-j 2pi f L_p / c). mmIR uses Monte-Carlo shooting-bouncing
rays + specular-manifold sampling; the image method is the fast specular core of that.

This module is numpy (feasibility test); a differentiable torch + CUDA version follows once
it is shown to reproduce real WiFi structure.
"""
import numpy as np

C_LIGHT = 299792458.0


def fit_planes(points, n_planes=6, dist=0.08, min_inliers=500):
    """RANSAC plane segmentation of the scaffold -> list of dict(n, d, centroid, pts)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(points)
    planes = []
    for _ in range(n_planes):
        if len(pc.points) < min_inliers:
            break
        model, inl = pc.segment_plane(dist, 3, 500)
        if len(inl) < min_inliers:
            break
        a, b, c, d = model; n = np.array([a, b, c]); nn = np.linalg.norm(n)
        ipts = np.asarray(pc.points)[inl]
        planes.append({"n": n / nn, "d": d / nn, "centroid": ipts.mean(0),
                       "lo": ipts.min(0), "hi": ipts.max(0), "n_inl": len(inl)})
        pc = pc.select_by_index(inl, invert=True)
    return planes


def reflect(S, pl):
    """Mirror image of point(s) S across plane n.x + d = 0."""
    return S - 2.0 * (S @ pl["n"] + pl["d"])[..., None] * pl["n"]


def seg_plane_hit(A, Bp, pl, pad=0.4):
    """Where segment A->Bp crosses the plane; returns (point, valid) with extent check."""
    nA, nB = A @ pl["n"] + pl["d"], Bp @ pl["n"] + pl["d"]
    if nA * nB > 0:                                 # same side -> no crossing
        return None, False
    t = nA / (nA - nB + 1e-12)
    P = A + t * (Bp - A)
    inside = np.all(P >= pl["lo"] - pad) and np.all(P <= pl["hi"] + pad)
    return P, inside


def fresnel_gamma(eps_r, cos_i):
    """Magnitude+phase of the TE Fresnel reflection coefficient (real eps approx)."""
    cos_i = np.clip(cos_i, 1e-3, 1.0)
    s = np.sqrt(np.clip(eps_r - (1 - cos_i ** 2), 1e-9, None))
    return (cos_i - s) / (cos_i + s + 1e-12)


def link_paths(tx, rx, planes, eps_per_plane, order=2):
    """Return list of (length, complex_gain) for LOS + 1st + 2nd order specular paths."""
    out = []
    L0 = np.linalg.norm(tx - rx); out.append((L0, 1.0 / (L0 + 1e-9)))      # LOS
    for k, pk in enumerate(planes):
        imgk = reflect(tx, pk)
        P, ok = seg_plane_hit(imgk, rx, pk)
        if not ok:
            continue
        L = np.linalg.norm(imgk - rx)
        cos_i = abs((rx - P) @ pk["n"]) / (np.linalg.norm(rx - P) + 1e-9)
        g = fresnel_gamma(eps_per_plane[k], cos_i) / (L + 1e-9)
        out.append((L, g))
        if order >= 2:
            for j, pj in enumerate(planes):
                if j == k:
                    continue
                imgkj = reflect(imgk, pj)
                P2, ok2 = seg_plane_hit(imgkj, rx, pj)
                if not ok2:
                    continue
                P1, ok1 = seg_plane_hit(imgk, P2, pk)
                if not ok1:
                    continue
                L2 = np.linalg.norm(imgkj - rx)
                ci2 = abs((rx - P2) @ pj["n"]) / (np.linalg.norm(rx - P2) + 1e-9)
                ci1 = abs((P2 - P1) @ pk["n"]) / (np.linalg.norm(P2 - P1) + 1e-9)
                g2 = (fresnel_gamma(eps_per_plane[k], ci1) *
                      fresnel_gamma(eps_per_plane[j], ci2)) / (L2 + 1e-9)
                out.append((L2, g2))
    return out


def link_paths_geom(tx, rx, planes, order=2):
    """Material-INDEPENDENT path geometry for one Tx->Rx link: list of
    (length, [(plane_idx, cos_incidence), ...]). Empty bounce list = LOS.
    This is the (non-differentiable) geometry; materials are applied later (differentiable).
    Mirror formula matches mmIR renderer/specular/image_method.py exactly."""
    out = []
    out.append((np.linalg.norm(tx - rx), []))                              # LOS
    for k, pk in enumerate(planes):
        imgk = reflect(tx, pk)
        P, ok = seg_plane_hit(imgk, rx, pk)
        if not ok:
            continue
        L = np.linalg.norm(imgk - rx)
        ci = abs((rx - P) @ pk["n"]) / (np.linalg.norm(rx - P) + 1e-9)
        out.append((L, [(k, ci)]))
        if order >= 2:
            for j, pj in enumerate(planes):
                if j == k:
                    continue
                imgkj = reflect(imgk, pj)
                P2, ok2 = seg_plane_hit(imgkj, rx, pj)
                if not ok2:
                    continue
                P1, ok1 = seg_plane_hit(imgk, P2, pk)
                if not ok1:
                    continue
                L2 = np.linalg.norm(imgkj - rx)
                ci2 = abs((rx - P2) @ pj["n"]) / (np.linalg.norm(rx - P2) + 1e-9)
                ci1 = abs((P2 - P1) @ pk["n"]) / (np.linalg.norm(P2 - P1) + 1e-9)
                out.append((L2, [(k, ci1), (j, ci2)]))
    return out


def render_csi(tx, rx_list, planes, eps_per_plane, sc_hz, order=2):
    """Complex CSI (n_links, n_sc) for 1 Tx and a list of Rx antenna positions."""
    H = np.zeros((len(rx_list), len(sc_hz)), dtype=np.complex128)
    for i, rx in enumerate(rx_list):
        for L, g in link_paths(tx, rx, planes, eps_per_plane, order):
            H[i] += g * np.exp(-1j * 2 * np.pi * sc_hz * L / C_LIGHT)
    return H
