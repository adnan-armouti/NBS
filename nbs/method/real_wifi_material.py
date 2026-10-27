"""Real-data material recovery from XRF55 WiFi CSI via the verified differentiable multibounce.

Fit per-plane permittivity to the measured |CSI| (9 links x 30 subcarriers) of the room, with
the corrected (leveled) geometry. Report fit before (default materials) / wrong-material floor /
after (recovered), plus the recovered eps (physical?). This is the WiFi side of mmWave<->WiFi
translation, now runnable because (a) multibounce renders WiFi multipath and (b) the camera
pose was leveled.

  python -m method.real_wifi_material --scene Scene1 --index 0
"""
import os, sys, argparse, glob, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import multibounce as MB, multibounce_torch as MT
from preprocessing import xrf55_config as C

DEV = "cpu"


def corr(x, y):
    x, y = x.flatten() - x.mean(), y.flatten() - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()

    # room box: use the WiFi-calibrated box if available, else the raw scaffold extent
    boxp = os.path.join(C.NBS_DATA_OUT, a.scene, "wifi_room_box.npy")
    if os.path.exists(boxp):
        e = np.load(boxp); lo = np.array([e[0], e[2], e[4]]); hi = np.array([e[1], e[3], e[5]])
        print(f"[geometry] using WiFi-calibrated room box")
    else:
        p = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy"))[:, :3]
        lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
        print(f"[geometry] using raw scaffold extent (run wifi_geom_calib for the calibrated box)")
    def plane(n, pt): n = np.array(n, float); return {"n": n, "d": -n @ pt, "lo": np.full(3, -99.), "hi": np.full(3, 99.)}
    planes = [plane([0, 0, 1], [0, 0, lo[2]]), plane([0, 0, -1], [0, 0, hi[2]]),
              plane([1, 0, 0], [lo[0], 0, 0]), plane([-1, 0, 0], [hi[0], 0, 0]),
              plane([0, 1, 0], [0, lo[1], 0]), plane([0, -1, 0], [0, hi[1], 0])]
    names = ["floor", "ceiling", "wall-x-", "wall-x+", "wall-y-", "wall-y+"]

    # WiFi geometry + real CSI
    tx_c, rx_c = C.wifi_layout(); lam = 299792458.0 / (C.WIFI_FREQ_GHZ * 1e9); d = lam / 2
    tx = np.array(tx_c); rx_list = [np.array([cx + (k - 1) * d, cy, cz]) for cx, cy, cz in rx_c for k in range(3)]
    sc = (C.WIFI_FREQ_GHZ + (np.arange(30) - 14.5) * 40e6 / 30 / 1e9) * 1e9
    paths = MT.precompute_paths(tx, rx_list, planes, order=2)
    f = sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", a.scene, "WiFi", "*.npy")))[a.index]
    real = torch.tensor(np.median(np.load(f), 1).reshape(9, 30), dtype=torch.float64)

    def fit_gain(H):                                # least-squares global scale (unknown hw gain)
        g = (real * H).sum() / (H * H).sum().clamp(min=1e-12)
        return g * H

    # default materials & wrong-material FLOOR
    eps_def = torch.tensor([2.0, 2.7, 5.3, 5.3, 5.3, 5.3], dtype=torch.float64)
    H_def = MT.render_csi(paths, eps_def, sc).abs()
    H_floor = MT.render_csi(paths, torch.full((6,), 5.3, dtype=torch.float64), sc).abs()   # all-concrete
    print(f"[default materials] per-link corr={corr(fit_gain(H_def).mean(1), real.mean(1)):.3f} full={corr(fit_gain(H_def), real):.3f}")
    print(f"[wrong-material floor (all-concrete)] full corr={corr(fit_gain(H_floor), real):.3f}")

    # recover per-plane material as a soft assignment over a PHYSICAL ITU library
    # (prevents the degenerate eps->1 collapse; recovers which material each surface is)
    LIB = torch.tensor([2.0, 2.7, 5.3, 6.3, 7.1], dtype=torch.float64)  # wood/plaster/concrete/glass/marble
    LIB_NAMES = ["wood", "plaster", "concrete", "glass", "marble"]
    logit = torch.zeros(6, len(LIB), requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.1)
    for _ in range(800):
        opt.zero_grad()
        eps = (torch.softmax(logit, 1) * LIB).sum(1)
        H = MT.render_csi(paths, eps, sc).abs()
        Hs = fit_gain(H)
        loss = ((Hs - real) ** 2).sum() / (real ** 2).sum()
        loss.backward(); opt.step()
    eps = ((torch.softmax(logit, 1) * LIB).sum(1)).detach()
    ident = [LIB_NAMES[i] for i in torch.softmax(logit, 1).argmax(1)]
    H = MT.render_csi(paths, eps, sc).abs(); Hs = fit_gain(H)
    print(f"\n[recovered materials] per-link corr={corr(Hs.mean(1), real.mean(1)):.3f} full={corr(Hs, real):.3f}")
    print(f"  identified material per surface:")
    for n, idn, e in zip(names, ident, eps.numpy()):
        print(f"    {n:9s} -> {idn:9s} (eps={e:.1f})")
    print(f"  (expected: floor~wood, ceiling~plaster, walls~concrete, window-wall~glass)")


if __name__ == "__main__":
    main()
