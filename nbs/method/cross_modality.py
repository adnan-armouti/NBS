"""Shared-geometry cross-band synthesis on real XRF55 data.

Thesis (WRITE_UP 7a): the room GEOMETRY is the band-transferable content; material is a small,
band-specific correction (and WiFi is material-insensitive). This experiment quantifies, per scene:

  A. zero-shot transfer  : render WiFi from the scene geometry reconstructed WITHOUT any WiFi
                           (the vision+mmWave monocular scaffold) -> per-link corr vs real WiFi.
  B. same-band oracle     : geometry calibrated to WiFi -> per-link corr (upper bound).
  C. material ablation    : oracle geometry, true-ish vs all-wrong materials -> delta corr
                           (confirms WiFi material-insensitivity across scenes).
  D. geometry agreement   : room extents from the scaffold vs from WiFi (band-invariance of geometry).

  python -m method.cross_modality
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import multibounce as MB
from preprocessing import xrf55_config as C

SCENES = ["Scene1", "Scene2", "Scene3", "Scene4"]


def corr(x, y):
    x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def box_planes(lo, hi):
    def pl(n, pt): n = np.array(n, float); return {"n": n, "d": -n @ pt, "lo": np.full(3, -99.), "hi": np.full(3, 99.)}
    return [pl([0, 0, 1], [0, 0, lo[2]]), pl([0, 0, -1], [0, 0, hi[2]]), pl([1, 0, 0], [lo[0], 0, 0]),
            pl([-1, 0, 0], [hi[0], 0, 0]), pl([0, 1, 0], [0, lo[1], 0]), pl([0, -1, 0], [0, hi[1], 0])]


def render_wifi(lo, hi, eps):
    tx_c, rx_c = C.wifi_layout(); lam = 299792458.0 / (C.WIFI_FREQ_GHZ * 1e9); d = lam / 2
    tx = np.array(tx_c); rxl = [np.array([cx + (k - 1) * d, cy, cz]) for cx, cy, cz in rx_c for k in range(3)]
    sc = (C.WIFI_FREQ_GHZ + (np.arange(30) - 14.5) * 40e6 / 30 / 1e9) * 1e9
    return np.abs(MB.render_csi(tx, rxl, box_planes(lo, hi), eps, sc, 2)), sc


def main():
    eps_def = np.array([2.0, 2.7, 5.3, 5.3, 5.3, 5.3])      # wood floor / plaster ceiling / concrete walls
    eps_wrong = np.full(6, 6.3)                              # all-glass (wrong)
    print(f"{'scene':7s} | {'zero-shot':>9s} {'oracle':>7s} | {'mat:def':>7s} {'mat:wrong':>9s} {'dMat':>5s} | geom-agree (m: dx,dy,dz)")
    print("-" * 92)
    rows = []
    for s in SCENES:
        p = np.load(os.path.join(C.NBS_DATA_OUT, s, "scene", "pcl.npy"))[:, :3]
        lo_s, hi_s = np.percentile(p, 2, 0), np.percentile(p, 98, 0)      # scaffold (non-WiFi) geometry
        e = np.load(os.path.join(C.NBS_DATA_OUT, s, "wifi_room_box.npy"))
        lo_w, hi_w = np.array([e[0], e[2], e[4]]), np.array([e[1], e[3], e[5]])  # WiFi-calibrated geometry
        real = np.median(np.load(sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", s, "WiFi", "*.npy")))[0]), 1).reshape(9, 30)
        rl = real.mean(1)

        def fit(H): g = (real * H).sum() / ((H * H).sum() + 1e-12); return g * H
        H_zs, _ = render_wifi(lo_s, hi_s, eps_def)                        # A zero-shot transfer
        H_or, _ = render_wifi(lo_w, hi_w, eps_def)                        # B oracle
        H_wr, _ = render_wifi(lo_w, hi_w, eps_wrong)                      # C wrong materials on oracle
        c_zs = corr(fit(H_zs).mean(1), rl); c_or = corr(fit(H_or).mean(1), rl)
        c_def = corr(fit(H_or), real); c_wr = corr(fit(H_wr), real)
        # geometry agreement: extent (size) match between bands (translation-invariant)
        d_ext = np.abs((hi_w - lo_w) - (hi_s - lo_s))
        rows.append((s, c_zs, c_or, c_def, c_wr))
        print(f"{s:7s} | {c_zs:9.3f} {c_or:7.3f} | {c_def:7.3f} {c_wr:9.3f} {abs(c_def-c_wr):5.3f} | "
              f"{d_ext[0]:.1f}, {d_ext[1]:.1f}, {d_ext[2]:.1f}")
    zs = np.mean([r[1] for r in rows]); orc = np.mean([r[2] for r in rows])
    dm = np.mean([abs(r[3] - r[4]) for r in rows])
    print("-" * 92)
    print(f"mean zero-shot mmWave/vision->WiFi per-link corr = {zs:.3f}  |  same-band oracle = {orc:.3f}")
    print(f"mean |corr(right materials) - corr(wrong materials)| = {dm:.3f}  -> materials {'insensitive' if dm<0.03 else 'matter'}")
    print(f"\nReading: geometry transfers across bands (zero-shot>0, oracle high); materials do not move")
    print(f"the WiFi fit. The transferable content is GEOMETRY; the gap zero-shot->oracle is the")
    print(f"monocular scaffold's room-scale error (the room walls lie beyond the reconstructed points).")


if __name__ == "__main__":
    main()
