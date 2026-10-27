"""Is real-data mmWave->WiFi material translation even viable? The decisive question:
is the measured XRF55 WiFi CSI dominated by the direct Tx->Rx line-of-sight (material-
INDEPENDENT), or does it carry scene-reflection structure (material-DEPENDENT) our model
can recover and transfer?

We compute, for each of the 9 links:
  - LOS-only |H| (direct Tx->Rx path, flat across subcarriers, material-independent)
  - measured |CSI| (static, median over time), per subcarrier
and report how frequency-selective the real CSI is (selectivity = reflection/multipath content
beyond LOS). High selectivity => material content exists; flat => LOS-dominated => translation moot.

  python -m method.wifi_real_check --scene Scene1 --index 0
"""
import os, sys, argparse, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import wifi_forward as WF
from method.freq_material import C_LIGHT
from preprocessing import xrf55_config as C

DEV = "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    import glob
    f = sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", a.scene, "WiFi", "*.npy")))[a.index]
    raw = np.load(f)                                  # (270, 1000) amplitude |CSI|
    print(f"[real WiFi] {os.path.basename(f)} shape={raw.shape} min={raw.min():.3g} max={raw.max():.3g}")
    csi = np.median(raw, axis=1).reshape(3, 3, 30)    # (rx, ant, sub) static |CSI|
    csi = csi.reshape(9, 30)                           # 9 links x 30 subcarriers

    # LOS |H| per link: 1/d_txrx (flat across subcarriers)
    tx, rx = WF.real_wifi_antennas(DEV)
    d = (rx - tx).norm(dim=1).numpy()                  # (9,) link distances
    print(f"[geometry] Tx->Rx link distances (m): {d.round(2)}")
    los = (1.0 / d)[:, None].repeat(30, axis=1)        # (9,30)

    # frequency selectivity of the REAL CSI per link (std/mean across subcarriers)
    sel = (csi.std(axis=1) / (csi.mean(axis=1) + 1e-9))
    print(f"\n[selectivity] per-link |CSI| subcarrier std/mean: {sel.round(2)}")
    print(f"  mean selectivity = {sel.mean():.2f}  (>~0.2 => real multipath/reflection structure; ~0 => flat/LOS-like)")

    # how well does LOS-only (geometry, material-independent) explain the real CSI across links?
    def corr(x, y):
        x, y = x.flatten() - x.mean(), y.flatten() - y.mean()
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    # per-link mean magnitude: does 1/d explain which links are strong?
    link_mag = csi.mean(axis=1)
    print(f"\n[LOS explains link strength?] corr(real per-link |CSI| , 1/d) = {corr(link_mag, 1/d):.3f}")
    print("  (high => the dominant structure is just the LOS path geometry, material-independent)")

    # ---- render WiFi from the scaffold (scene reflections + LOS) and correlate with real ----
    from method import ra_forward as RF
    from method.freq_material import eps_from_dispersion, ITU_GT
    p = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy"))
    ii = np.random.RandomState(0).choice(len(p), 3000, replace=False); p = p[ii]
    pos = torch.tensor(p[:, :3], dtype=torch.float64); nrm = torch.tensor(p[:, 3:6], dtype=torch.float64)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64)
    sh = torch.full((pos.shape[0],), 3e-4, dtype=torch.float64)
    abcd = torch.stack([torch.tensor(ITU_GT["concrete"], dtype=torch.float64)] * pos.shape[0])
    amp = RF._amp_from_eps(*eps_from_dispersion(abcd, C.WIFI_FREQ_GHZ), thick, sh, pos, nrm, C.WIFI_FREQ_GHZ)
    sc = WF.subcarrier_ghz(C.WIFI_FREQ_GHZ, 40.0, 30)
    H_scene = WF.render_csi(amp, pos, tx, rx, sc)                      # (9,30) complex (scene reflections)
    fsc = torch.tensor(sc, dtype=torch.float64) * 1e9
    dd = torch.tensor(d)
    H_los = (1.0 / dd)[:, None] * torch.exp(-1j * 2 * math.pi * fsc[None, :] * dd[:, None] / C_LIGHT)
    for w in [0.0, 1.0, 5.0]:                                          # vary LOS weight
        H = (H_scene + w * H_los).abs().numpy()
        print(f"[render vs real |CSI|] LOS weight={w:.0f}: corr={corr(H, csi):.3f}")
    print("  (compare to the mmWave fit of 0.76; low here => single-bounce model cannot reproduce real WiFi multipath)")


if __name__ == "__main__":
    main()
