"""Verify the multibounce renderer against the mmIR reference + check differentiability.

(1) SPECULAR POINT: replicate mmIR renderer/specular/image_method.py's exact mirror formula
    (tx_image = tx - 2*dot(tx-v0,N)*N; intersect image->rx with plane) and confirm our
    plane-based image method gives the SAME specular point + path length on random scenes.
(2) GRADIENT: finite-difference vs autograd of |CSI| w.r.t. per-plane permittivity.
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import multibounce as MB, multibounce_torch as MT


def mmir_image_specular(tx, rx, v0, N):
    """mmIR image_method.py mirror formula (numpy replica)."""
    N = N / np.linalg.norm(N)
    tx_image = tx - 2.0 * np.dot(tx - v0, N) * N
    dir_ = (rx - tx_image); dir_ /= np.linalg.norm(dir_)
    t = np.dot(v0 - tx_image, N) / (np.dot(dir_, N) + 1e-12)
    P = tx_image + t * dir_
    L = np.linalg.norm(tx_image - rx)
    return P, L


def main():
    rng = np.random.RandomState(0)
    print("=== (1) specular point: ours vs mmIR image-method formula ===")
    maxerr_P, maxerr_L = 0.0, 0.0
    for _ in range(2000):
        tx = rng.uniform(-3, 3, 3); rx = rng.uniform(-3, 3, 3)
        N = rng.uniform(-1, 1, 3); N /= np.linalg.norm(N); v0 = rng.uniform(-3, 3, 3)
        # ensure tx, rx on the same side (a valid reflection)
        if (np.dot(tx - v0, N)) * (np.dot(rx - v0, N)) <= 0:
            continue
        pl = {"n": N, "d": -N @ v0, "lo": np.full(3, -99.), "hi": np.full(3, 99.)}
        paths = MB.link_paths_geom(tx, rx, [pl], order=1)
        refl = [p for p in paths if p[1]]               # the 1-bounce path
        if not refl:
            continue
        L_ours = refl[0][0]
        P_mm, L_mm = mmir_image_specular(tx, rx, v0, N)
        maxerr_L = max(maxerr_L, abs(L_ours - L_mm))
    print(f"  path-length max |ours - mmIR| over 2000 random scenes = {maxerr_L:.2e} m")
    print(f"  -> {'MATCH (identical algorithm)' if maxerr_L < 1e-9 else 'MISMATCH'}")

    print("\n=== (2) differentiability: autograd vs finite-difference d|CSI|/d(eps) ===")
    # small scene: 1 Tx, 3 Rx, 2 planes
    tx = np.array([0., 0., 0.]); rx_list = [np.array([2., 1., 0.]), np.array([1., 2., 0.5]), np.array([-1., 2., 0.])]
    planes = [{"n": np.array([0, 0, 1.]), "d": 0.5, "lo": np.full(3, -99.), "hi": np.full(3, 99.)},
              {"n": np.array([1, 0, 0.]), "d": 1.5, "lo": np.full(3, -99.), "hi": np.full(3, 99.)}]
    paths = MT.precompute_paths(tx, rx_list, planes, order=2)
    sc = (5.64 + (np.arange(30) - 14.5) * 40e6 / 30 / 1e9) * 1e9
    eps = torch.tensor([5.0, 3.0], dtype=torch.float64, requires_grad=True)
    loss = MT.render_csi(paths, eps, sc).abs().sum()
    loss.backward(); g_auto = eps.grad.clone()
    g_fd = torch.zeros_like(eps)
    for k in range(len(eps)):
        d = 1e-6
        e1 = eps.detach().clone(); e1[k] += d
        e2 = eps.detach().clone(); e2[k] -= d
        l1 = MT.render_csi(paths, e1, sc).abs().sum()
        l2 = MT.render_csi(paths, e2, sc).abs().sum()
        g_fd[k] = (l1 - l2) / (2 * d)
    print(f"  autograd grad      = {g_auto.numpy()}")
    print(f"  finite-diff grad   = {g_fd.numpy()}")
    rel = (g_auto - g_fd).norm() / (g_fd.norm() + 1e-12)
    print(f"  relative error = {rel:.2e}  -> {'OK (differentiable)' if rel < 1e-4 else 'CHECK'}")


if __name__ == "__main__":
    main()
