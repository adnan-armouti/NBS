"""Calibrate the WiFi room geometry to the measured CSI (the WiFi analog of the mmWave
chirp calibration that took the RA fit 0.42->0.76). The monocular scaffold does not reach
the true walls, so the room box is mis-sized -> wrong path lengths. Optimize the 6 box
extents (+ a global delay) by maximizing corr(rendered |CSI|, real |CSI|), derivative-free.
Materials fixed; geometry is what limits the fit at this stage.

  python -m method.wifi_geom_calib --scene Scene1 --index 0
"""
import os, sys, argparse, glob
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import multibounce as MB
from preprocessing import xrf55_config as C


def corr(x, y):
    x, y = x.flatten() - x.mean(), y.flatten() - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    p = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy"))[:, :3]
    lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
    tx_c, rx_c = C.wifi_layout(); lam = 299792458.0 / (C.WIFI_FREQ_GHZ * 1e9); d = lam / 2
    tx = np.array(tx_c); rxl = [np.array([cx + (k - 1) * d, cy, cz]) for cx, cy, cz in rx_c for k in range(3)]
    sc = (C.WIFI_FREQ_GHZ + (np.arange(30) - 14.5) * 40e6 / 30 / 1e9) * 1e9
    real = np.median(np.load(sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", a.scene, "WiFi", "*.npy")))[a.index]), 1).reshape(9, 30)
    eps = np.array([2.0, 2.7, 5.3, 5.3, 5.3, 5.3])

    def box_from(ext):
        x0, x1, y0, y1, z0, z1 = ext
        def pl(n, pt): n = np.array(n, float); return {"n": n, "d": -n @ pt, "lo": np.full(3, -99.), "hi": np.full(3, 99.)}
        return [pl([0, 0, 1], [0, 0, z0]), pl([0, 0, -1], [0, 0, z1]), pl([1, 0, 0], [x0, 0, 0]),
                pl([-1, 0, 0], [x1, 0, 0]), pl([0, 1, 0], [0, y0, 0]), pl([0, -1, 0], [0, y1, 0])]

    def fit_corr(ext):
        H = np.abs(MB.render_csi(tx, rxl, box_from(ext), eps, sc, 2))
        g = (real * H).sum() / ((H * H).sum() + 1e-12)
        return corr(g * H, real)

    x0 = np.array([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
    print(f"[init box] x[{x0[0]:.1f},{x0[1]:.1f}] y[{x0[2]:.1f},{x0[3]:.1f}] z[{x0[4]:.1f},{x0[5]:.1f}]  corr={fit_corr(x0):.3f}")

    # coarse multi-start (the room walls may be beyond the scaffold extent)
    best = (fit_corr(x0), x0)
    rng = np.random.RandomState(0)
    for _ in range(40):
        cand = x0 + rng.uniform(-1.5, 1.5, 6) * np.array([1, 1, 1, 1, 0.5, 0.5])
        cand[1] = max(cand[1], cand[0] + 1); cand[3] = max(cand[3], cand[2] + 1); cand[5] = max(cand[5], cand[4] + 1)
        r = minimize(lambda e: -fit_corr(e), cand, method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 0.02, "fatol": 1e-3})
        if -r.fun > best[0]:
            best = (-r.fun, r.x)
    c, e = best
    print(f"[calibrated box] x[{e[0]:.1f},{e[1]:.1f}] y[{e[2]:.1f},{e[3]:.1f}] z[{e[4]:.1f},{e[5]:.1f}]")
    print(f"  WiFi per-link/full corr after geometry calibration: full={c:.3f}")
    H = np.abs(MB.render_csi(tx, rxl, box_from(e), eps, sc, 2)); g = (real * H).sum() / ((H * H).sum() + 1e-12)
    print(f"  per-link corr={corr((g*H).mean(1), real.mean(1)):.3f}  (was 0.45 with the uncalibrated box)")
    np.save(os.path.join(C.NBS_DATA_OUT, a.scene, "wifi_room_box.npy"), e)
    print(f"  saved calibrated box -> {a.scene}/wifi_room_box.npy")


if __name__ == "__main__":
    main()
