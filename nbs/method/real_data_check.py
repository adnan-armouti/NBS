"""Prerequisite for any REAL-data material recovery: does the single-bounce forward model
+ the monocular-depth scaffold reproduce the MEASURED XRF55 RA at 60 GHz at all?

Here correlation IS meaningful (the two sources are INDEPENDENT: a real radar measurement
vs our model of the room), unlike the cross-band case (same geometry both sides).

  python -m method.real_data_check --scene Scene1
"""
import os, sys, argparse, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF
from method.freq_material import eps_from_dispersion, ITU_GT
from preprocessing import xrf55_io as IO, xrf55_config as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# chirp params chosen so range_res ~ 0.076 m/bin (recovered XRF55 value), 256 range bins
S = 299792458.0 / (2 * 0.076) / 60e-6      # bandwidth/chirp_T -> S
CHIRP_T = 60e-6


def db_img(x):
    x = x.abs() if torch.is_complex(x) else x
    x = x / (x.max() + 1e-9)
    return torch.clamp(20 * torch.log10(x + 1e-6), -40, 0)


def corr(a, b):
    a, b = a.flatten().float(), b.flatten().float(); a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--npts", type=int, default=4000)
    a = ap.parse_args()

    mmw = IO.list_mmwave(a.scene)[a.index]
    real = torch.tensor(np.median(IO.load_mmwave(mmw)["ra"], axis=0), device=DEV)   # (256 range, 64 az), static
    print(f"[real RA] {a.scene} {os.path.basename(mmw)}  shape={tuple(real.shape)} (range x az), static (median over frames)")

    p = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy"))
    idx = np.random.RandomState(0).choice(len(p), min(a.npts, len(p)), replace=False); p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV); nrm = nrm/nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256*0.076); pos, nrm = pos[keep], nrm[keep]
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    abcd = torch.stack([torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=DEV)] * pos.shape[0])
    sh = torch.full((pos.shape[0],), 3e-4, dtype=torch.float64, device=DEV)
    sim = RF.render_ra_disp(abcd, thick, sh, pos, nrm, 60.0, S=S, K=256, chirp_T=CHIRP_T, n_az=64).T  # (256,64)
    print(f"[sim RA] scaffold render shape={tuple(sim.shape)}  (range x az)")

    c_raw = corr(real, sim.abs())
    c_db = corr(db_img(real), db_img(sim))
    print(f"\n  corr(real, sim |RA|)  = {c_raw:.3f}")
    print(f"  corr(real, sim) in dB = {c_db:.3f}")
    print("  (independent sources, so corr IS meaningful here. >~0.3-0.5 => model captures real structure.)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(db_img(real).cpu(), aspect="auto", origin="lower"); ax[0].set_title("MEASURED XRF55 RA (static, dB)")
    ax[1].imshow(db_img(sim).cpu(), aspect="auto", origin="lower"); ax[1].set_title("forward-model RA of scaffold (dB)")
    ov = torch.zeros(256, 64, 3); ov[..., 0] = (db_img(real).cpu()+40)/40; ov[..., 1] = (db_img(sim).cpu()+40)/40
    ax[2].imshow(ov, aspect="auto", origin="lower"); ax[2].set_title(f"overlay (R=real,G=sim) corr_dB={c_db:.2f}")
    for x in ax: x.set_xlabel("azimuth bin"); x.set_ylabel("range bin")
    out = os.path.join("output", "real_data_check"); os.makedirs(out, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{a.scene}_real_vs_sim.png"), dpi=95); plt.close(fig)
    print(f"\nsaved {out}/{a.scene}_real_vs_sim.png")


if __name__ == "__main__":
    main()
