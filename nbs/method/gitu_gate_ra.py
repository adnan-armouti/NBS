"""Hardened G-ITU gate: real 3DPS BSDF + single-chip (XRF55) array + |RA| supervision + noise.

Recover per-scatterer ITU dispersion (a,b,c,d) by back-propagating a NOISY |RA| loss over
training bands (RA via the repo's single-chip pipeline), then ANALYTICALLY CONTINUE to
held-out bands. Compare to a per-band-eps + interpolation baseline (the field-interpolation
analogue). |RA| carries the relative inter-antenna phase, so this exercises the actual NBS
claim, not single-antenna ADC.

  python -m method.gitu_gate_ra
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF, freq_material as FM

torch.manual_seed(0)
DEV = "cpu"
NOISE_DB = -20.0                      # measurement SNR ~20 dB
TRAIN = [10.0, 24.0, 40.0]
HELD = [30, 45, 60, 77, 100, 130, 160]   # spans interpolation -> far extrapolation (4x train max)

# ---- scene (known geometry) ----
MATS = ["concrete", "glass", "wood", "plasterboard", "brick", "marble"]
M = len(MATS)
POS = torch.tensor(np.stack([np.linspace(-1.5, 1.5, M), np.linspace(2.0, 5.0, M), np.zeros(M)], 1),
                   dtype=torch.float64, device=DEV)
NRM = -POS / POS.norm(dim=1, keepdim=True)
GT = torch.tensor([FM.ITU_GT[m] for m in MATS], dtype=torch.float64, device=DEV)   # (M,4)
THICK = torch.full((M,), 0.15, dtype=torch.float64, device=DEV)
SH = torch.full((M,), 3e-4, dtype=torch.float64, device=DEV)


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def mag(ra):
    return ra.abs()


def ra_corr(a, b):
    a = mag(a).flatten(); b = mag(b).flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def ra_db(pred, gt):
    return float(20 * torch.log10((mag(pred) - mag(gt)).norm() / (mag(gt).norm() + 1e-12)))


def main():
    # measured (noisy) training RA, and CLEAN ground-truth held-out RA
    gt_train = {f: RF.render_ra_disp(GT, THICK, SH, POS, NRM, f, noise_db=NOISE_DB, gen=gen(100 + int(f)))
                for f in TRAIN}
    gt_held = {h: RF.render_ra_disp(GT, THICK, SH, POS, NRM, float(h)) for h in HELD}

    # ---------- NBS: recover dispersion (a,b,c,d) from noisy |RA| over train bands ----------
    init = GT.clone(); init[:, 0] *= 1.5; init[:, 2] *= 0.5; init[:, 3] *= 1.2
    p = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.03)
    lo = torch.tensor([1.0, -1.0, 1e-5, 0.0], device=DEV)
    for it in range(700):
        opt.zero_grad()
        loss = 0.0
        for f in TRAIN:
            pred = RF.render_ra_disp(p, THICK, SH, POS, NRM, f)
            g = gt_train[f]
            loss = loss + ((mag(pred) / mag(pred).norm() - mag(g) / mag(g).norm()) ** 2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            p.clamp_(min=lo)
    p = p.detach()

    # ---------- baseline: per-band complex eps (fit noisy |RA|), then interpolate eps ----------
    band_er, band_ei = {}, {}
    for f in TRAIN:
        er = (GT[:, 0] * 1.5).clone().requires_grad_(True)
        ei = (17.98 * GT[:, 2] * f ** (GT[:, 3] - 1) * 0.5).clone().requires_grad_(True)
        o = torch.optim.Adam([er, ei], lr=0.03)
        g = gt_train[f]
        for it in range(700):
            o.zero_grad()
            pred = RF.render_ra_eps(er, ei, THICK, SH, POS, NRM, f)
            loss = ((mag(pred) / mag(pred).norm() - mag(g) / mag(g).norm()) ** 2).sum()
            loss.backward(); o.step()
            with torch.no_grad():
                er.clamp_(min=1.0); ei.clamp_(min=1e-6)
        band_er[f] = er.detach(); band_ei[f] = ei.detach()
    fs = np.array(TRAIN)
    ER = torch.stack([band_er[f] for f in TRAIN]).cpu().numpy()   # (B,M)
    EI = torch.stack([band_ei[f] for f in TRAIN]).cpu().numpy()

    def interp_eps(h):
        er_h = np.array([np.interp(h, fs, ER[:, m]) if fs.min() <= h <= fs.max()
                         else np.polyval(np.polyfit(fs, ER[:, m], 1), h) for m in range(M)])
        ei_h = np.array([np.interp(h, fs, EI[:, m]) if fs.min() <= h <= fs.max()
                         else np.polyval(np.polyfit(fs, EI[:, m], 1), h) for m in range(M)])
        er_h = np.clip(er_h, 1.0, None); ei_h = np.clip(ei_h, 1e-6, None)  # keep physical
        return (torch.tensor(er_h, dtype=torch.float64, device=DEV),
                torch.tensor(ei_h, dtype=torch.float64, device=DEV))

    # ---------- sweep held-out bands ----------
    rows = []
    for h in HELD:
        nbs_ra = RF.render_ra_disp(p, THICK, SH, POS, NRM, float(h))
        er_h, ei_h = interp_eps(float(h))
        base_ra = RF.render_ra_eps(er_h, ei_h, THICK, SH, POS, NRM, float(h))
        r = {"held": h,
             "nbs_corr": ra_corr(nbs_ra, gt_held[h]), "nbs_db": ra_db(nbs_ra, gt_held[h]),
             "base_corr": ra_corr(base_ra, gt_held[h]), "base_db": ra_db(base_ra, gt_held[h])}
        rows.append(r)
        print(f"  held {h:>4} GHz | NBS corr={r['nbs_corr']:.4f} dB={r['nbs_db']:6.1f} "
              f"| interp corr={r['base_corr']:.4f} dB={r['base_db']:6.1f}")

    out = os.path.join("output", "gitu_gate_ra"); os.makedirs(out, exist_ok=True)
    json.dump({"train_bands": TRAIN, "noise_db": NOISE_DB, "rows": rows,
               "scene": {"materials": MATS, "n_scatterers": M}}, open(os.path.join(out, "results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = [r["held"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(h, [r["nbs_db"] for r in rows], "o-", lw=2, label="NBS (analytic ITU dispersion)")
    ax[0].plot(h, [r["base_db"] for r in rows], "s--", lw=2, label="per-band eps + interpolation")
    ax[0].axvspan(min(TRAIN), max(TRAIN), color="green", alpha=0.07, label="train-band span")
    ax[0].set_xlabel("held-out carrier (GHz)"); ax[0].set_ylabel("|RA| error (dB)")
    ax[0].set_title("Hardened G-ITU: |RA| fidelity vs band gap"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(h, [r["nbs_corr"] for r in rows], "o-", lw=2, label="NBS")
    ax[1].plot(h, [r["base_corr"] for r in rows], "s--", lw=2, label="interp")
    ax[1].set_xlabel("held-out carrier (GHz)"); ax[1].set_ylabel("|RA| Pearson corr"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("|RA| correlation vs band gap")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ra_band_gap_sweep.png"), dpi=110); plt.close(fig)
    print(f"\n[real BSDF + single-chip array + {NOISE_DB}dB noise + |RA|] saved {out}/")


if __name__ == "__main__":
    main()
