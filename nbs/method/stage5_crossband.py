"""Stage 5: cross-band inverse-rendering core on the REAL XRF55 scaffold geometry.

Scales the validated single-chip RA forward model + dispersion/roughness recovery from
point-scatterer toys to a real room (the Scene1 static scaffold, ~10^5 points). The room
is rendered at several carriers; per-material-class physics (dispersion a,b,c,d + roughness
sigma_h) is recovered from the training bands and used to SYNTHESIZE a held-out band, which
we visualise and compare to ground truth and to the XBandGS-style interpolation baseline.

HONEST SCOPE: XRF55 mmWave is single-band + processed magnitude, so real multi-band raw
validation is impossible on it (blocked on the R1 capture). This is Track A (controlled
materials/signal) on REAL geometry: it demonstrates the core loop at realistic scale and
produces a synthesized held-out-band RA of a real room.

  python -m method.stage5_crossband
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF, freq_material as FM

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCAFFOLD = "/home/adnan/Desktop/NBS/data/XRF55/Scene1/scene/pcl.npy"
N_POINTS = 2000
TRAIN = [24.0, 40.0, 60.0]
HELD = 90.0                                  # into the rough regime
NOISE_DB = -20.0

# 4 material classes by height (floor/low/mid/high) -> ITU materials + roughness
CLASS_MATS = ["wood", "plasterboard", "concrete", "glass"]
CLASS_SH = torch.tensor([8e-4, 5e-4, 3e-4, 1e-4], dtype=torch.float64)   # floor rough -> glass smooth


def load_scene():
    p = np.load(SCAFFOLD)
    idx = np.random.RandomState(0).choice(len(p), min(N_POINTS, len(p)), replace=False)
    p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    # keep points in the radar's usable range
    r = pos.norm(dim=1)
    keep = (r >= 0.5) & (r < 256 * 0.076)
    pos, nrm = pos[keep], nrm[keep]
    # class by z-height quartiles
    z = pos[:, 2]
    q = torch.quantile(z, torch.tensor([0.25, 0.5, 0.75], device=DEV, dtype=z.dtype))
    cls = (z[:, None] > q[None, :]).sum(1)        # 0..3
    return pos, nrm, cls


def gen(s): return torch.Generator(device=DEV).manual_seed(s)


def expand(class_param, cls):
    """(K,...) per-class -> (N,...) per-point via class index."""
    return class_param[cls]


def main():
    pos, nrm, cls = load_scene()
    N = pos.shape[0]
    print(f"[scene] {N} points, classes {[int((cls==k).sum()) for k in range(4)]}")
    gt_abcd_cls = torch.tensor([FM.ITU_GT[m] for m in CLASS_MATS], dtype=torch.float64, device=DEV)
    gt_sh_cls = CLASS_SH.to(DEV)
    thick = torch.full((N,), 0.15, dtype=torch.float64, device=DEV)
    abcd_pt = expand(gt_abcd_cls, cls)            # (N,4)
    sh_pt = expand(gt_sh_cls, cls)                # (N,)

    def render(abcd, sh, f, noise=None, g=None):
        return RF.render_ra_disp(abcd, thick, sh, pos, nrm, f, K=256, n_az=64, noise_db=noise, gen=g)

    gt_tr = {f: render(abcd_pt, sh_pt, f, NOISE_DB, gen(100 + int(f))) for f in TRAIN}
    gt_hd = render(abcd_pt, sh_pt, HELD)

    # ---- NBS: recover per-CLASS dispersion + roughness ----
    a = gt_abcd_cls.clone(); a[:, 0] *= 1.4; a[:, 2] *= 0.6; a[:, 3] *= 1.15
    a = a.clone().requires_grad_(True)
    lsh = torch.log(gt_sh_cls * 2.0).clone().requires_grad_(True)
    opt = torch.optim.Adam([a, lsh], lr=0.03)
    lo = torch.tensor([1.0, -1.0, 1e-5, 0.0], device=DEV)
    for it in range(500):
        opt.zero_grad()
        ap, shp = expand(a, cls), expand(torch.exp(lsh), cls)
        loss = sum((((render(ap, shp, f).abs() - gt_tr[f].abs()) ** 2).sum()
                    / (gt_tr[f].abs() ** 2).sum()) for f in TRAIN)
        loss.backward(); opt.step()
        with torch.no_grad():
            a.clamp_(min=lo); lsh.clamp_(min=math.log(1e-5), max=math.log(5e-3))
        if it % 100 == 0:
            print(f"  it{it:03d} loss={loss.item():.4f}")
    a = a.detach(); sh_rec = torch.exp(lsh.detach())
    nbs_hd = render(expand(a, cls), expand(sh_rec, cls), HELD)

    # ---- interp baseline: per-band per-class complex reflectivity, interpolate ----
    # recover one complex reflectivity scale per class per band (fit |RA|)
    band_amp = {}
    for f in TRAIN:
        amp = torch.complex(torch.ones(4, device=DEV, dtype=torch.float64) * 0.1,
                            torch.zeros(4, device=DEV, dtype=torch.float64)).clone()
        ar = amp.real.clone().requires_grad_(True); ai = amp.imag.clone().requires_grad_(True)
        o = torch.optim.Adam([ar, ai], lr=0.02)
        for it in range(400):
            o.zero_grad()
            amp_pt = expand(torch.complex(ar, ai), cls)
            pred = RF._render_ra(amp_pt.to(torch.complex128), pos, f, 60e12, 256, 60e-6, 64, None, None, DEV)
            loss = ((pred.abs() - gt_tr[f].abs()) ** 2).sum() / (gt_tr[f].abs() ** 2).sum()
            loss.backward(); o.step()
        band_amp[f] = torch.complex(ar.detach(), ai.detach())
    fs = np.array(TRAIN)
    AR = torch.stack([band_amp[f].real for f in TRAIN]).cpu().numpy()
    AI = torch.stack([band_amp[f].imag for f in TRAIN]).cpu().numpy()
    ar_h = np.array([np.polyval(np.polyfit(fs, AR[:, k], 1), HELD) for k in range(4)])
    ai_h = np.array([np.polyval(np.polyfit(fs, AI[:, k], 1), HELD) for k in range(4)])
    amp_h = torch.complex(torch.tensor(ar_h, device=DEV), torch.tensor(ai_h, device=DEV))
    base_hd = RF._render_ra(expand(amp_h, cls).to(torch.complex128), pos, HELD, 60e12, 256, 60e-6, 64, None, None, DEV)

    # ---- metrics ----
    def corr(p, g):
        p, g = p.abs().flatten(), g.abs().flatten(); p, g = p - p.mean(), g - g.mean()
        return float((p * g).sum() / (p.norm() * g.norm() + 1e-12))
    def db(p, g):
        return float(20 * torch.log10((p.abs() - g.abs()).norm() / (g.abs().norm() + 1e-12)))
    res = {"held_ghz": HELD, "train": TRAIN, "n_points": N,
           "nbs_corr": corr(nbs_hd, gt_hd), "nbs_db": db(nbs_hd, gt_hd),
           "interp_corr": corr(base_hd, gt_hd), "interp_db": db(base_hd, gt_hd),
           "sigma_h_recovery_err_pct": (100 * (sh_rec - gt_sh_cls).abs() / gt_sh_cls).tolist()}
    print(f"\n[held {HELD} GHz] NBS corr={res['nbs_corr']:.3f} dB={res['nbs_db']:.1f} | "
          f"interp corr={res['interp_corr']:.3f} dB={res['interp_db']:.1f}")
    out = os.path.join("output", "stage5_crossband"); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    imgs = [("GT held-out RA", gt_hd), ("NBS synthesized", nbs_hd), ("interp synthesized", base_hd)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (t, ra) in zip(axs, imgs):
        ax.imshow(20 * torch.log10(ra.abs().cpu() + 1e-6), aspect="auto", origin="lower")
        ax.set_title(t); ax.set_xlabel("azimuth bin"); ax.set_ylabel("range bin")
    fig.suptitle(f"Stage 5: held-out {HELD} GHz RA of the real XRF55 room (train {TRAIN} GHz) [dB]")
    fig.tight_layout(); fig.savefig(os.path.join(out, "held_band_ra.png"), dpi=100); plt.close(fig)
    print(f"saved {out}/held_band_ra.png")


if __name__ == "__main__":
    main()
