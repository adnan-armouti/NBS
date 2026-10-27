"""Roughness regime-change test: where physics-recovery SHOULD beat field interpolation.

The Rayleigh roughness factor exp(-0.5*(4*pi*sigma_h*cos/lambda)^2) makes the coherent
reflectivity roll off NONLINEARLY as the surface crosses 4*pi*sigma_h/lambda ~ 1
(smooth -> rough). We TRAIN on low bands (surfaces smooth) and PREDICT high bands
(surfaces rough), i.e. the held-out bands are on the far side of the crossing.

  NBS    : recover physical (a,b,c,d) dispersion AND sigma_h, re-render via the physical
           forward model -> predicts the rolloff.
  interp : recover the per-scatterer COMPLEX reflectivity *attribute* per training band
           (the XBandGS-style learned field attribute), interpolate it across frequency,
           render with geometry at the held band -> cannot model the nonlinear rolloff.

Both fit the COMPLEX RA (fair: same observability); we evaluate at held-out bands.
  python -m method.gitu_roughness
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF, freq_material as FM

torch.manual_seed(0)
DEV = "cpu"
NOISE_DB = -20.0
TRAIN = [10.0, 18.0, 28.0]                 # low bands -> surfaces smooth
HELD = [28, 40, 60, 90, 120, 160]          # cross into the rough regime

MATS = ["concrete", "glass", "wood", "plasterboard", "brick", "marble"]
M = len(MATS)
POS = torch.tensor(np.stack([np.linspace(-1.5, 1.5, M), np.linspace(2.0, 5.0, M), np.zeros(M)], 1),
                   dtype=torch.float64, device=DEV)
NRM = -POS / POS.norm(dim=1, keepdim=True)
GT = torch.tensor([FM.ITU_GT[m] for m in MATS], dtype=torch.float64, device=DEV)
THICK = torch.full((M,), 0.15, dtype=torch.float64, device=DEV)
# roughness straddling the crossing: 4*pi*sigma_h/lambda ~ 1 around 50-90 GHz
SH = torch.tensor([2e-4, 4e-4, 6e-4, 3e-4, 5e-4, 7e-4], dtype=torch.float64, device=DEV)


def gen(s): return torch.Generator(device=DEV).manual_seed(s)
def mag(ra): return ra.abs()


def cx_corr(a, b):
    a = a.flatten(); b = b.flatten(); a = a - a.mean(); b = b - b.mean()
    return float((a.conj() * b).sum().abs() / (torch.sqrt((a.conj()*a).sum().real*(b.conj()*b).sum().real) + 1e-12))


def ra_db(pred, gt):
    return float(20 * torch.log10((mag(pred) - mag(gt)).norm() / (mag(gt).norm() + 1e-12)))


def cx_loss(pred, gt):
    return ((pred - gt).abs() ** 2).sum() / (gt.abs() ** 2).sum()


def main():
    # crossing diagnostic
    print("4*pi*sigma_h/lambda (smooth<1<rough):")
    for f in [TRAIN[0], TRAIN[-1], HELD[-1]]:
        lam = FM.C_LIGHT / (f * 1e9)
        print(f"  {f:>4} GHz: {(4*math.pi*SH/lam).cpu().numpy().round(2)}")

    gt_tr = {f: RF.render_ra_disp(GT, THICK, SH, POS, NRM, f, noise_db=NOISE_DB, gen=gen(100+int(f))) for f in TRAIN}
    gt_hd = {h: RF.render_ra_disp(GT, THICK, SH, POS, NRM, float(h)) for h in HELD}

    # ---- NBS: recover dispersion (a,b,c,d) + roughness sigma_h ----
    p = GT.clone(); p[:, 0] *= 1.4; p[:, 2] *= 0.6; p[:, 3] *= 1.15
    p = p.clone().requires_grad_(True)
    log_sh = torch.log(SH * 2.0).clone().requires_grad_(True)   # wrong init
    opt = torch.optim.Adam([p, log_sh], lr=0.03)
    lo = torch.tensor([1.0, -1.0, 1e-5, 0.0], device=DEV)
    for it in range(900):
        opt.zero_grad()
        sh = torch.exp(log_sh)
        loss = sum(cx_loss(RF.render_ra_disp(p, THICK, sh, POS, NRM, f), gt_tr[f]) for f in TRAIN)
        loss.backward(); opt.step()
        with torch.no_grad():
            p.clamp_(min=lo); log_sh.clamp_(min=math.log(1e-5), max=math.log(5e-3))
    p = p.detach(); sh_rec = torch.exp(log_sh.detach())

    # ---- interp baseline: per-band complex reflectivity attribute, interpolate ----
    band_amp = {}
    for f in TRAIN:
        amp = (RF._amp_from_eps(*FM.eps_from_dispersion(GT, f), THICK, SH, POS, NRM, f) * 0.6).clone()
        amp_r = amp.real.clone().requires_grad_(True); amp_i = amp.imag.clone().requires_grad_(True)
        o = torch.optim.Adam([amp_r, amp_i], lr=0.02)
        for it in range(900):
            o.zero_grad()
            a = torch.complex(amp_r, amp_i)
            pred = RF._render_ra(a, POS, f, 60e12, 128, 60e-6, 64, None, None, DEV)
            loss = cx_loss(pred, gt_tr[f]); loss.backward(); o.step()
        band_amp[f] = torch.complex(amp_r.detach(), amp_i.detach())
    fs = np.array(TRAIN)
    AR = torch.stack([band_amp[f].real for f in TRAIN]).cpu().numpy()   # (B,M)
    AI = torch.stack([band_amp[f].imag for f in TRAIN]).cpu().numpy()

    def interp_amp(h):
        ar = np.array([np.interp(h, fs, AR[:, m]) if fs.min() <= h <= fs.max()
                       else np.polyval(np.polyfit(fs, AR[:, m], 1), h) for m in range(M)])
        ai = np.array([np.interp(h, fs, AI[:, m]) if fs.min() <= h <= fs.max()
                       else np.polyval(np.polyfit(fs, AI[:, m], 1), h) for m in range(M)])
        return torch.complex(torch.tensor(ar, device=DEV), torch.tensor(ai, device=DEV))

    rows = []
    for h in HELD:
        nbs = RF.render_ra_disp(p, THICK, sh_rec, POS, NRM, float(h))
        base = RF._render_ra(interp_amp(float(h)), POS, float(h), 60e12, 128, 60e-6, 64, None, None, DEV)
        r = {"held": h, "nbs_cxcorr": cx_corr(nbs, gt_hd[h]), "nbs_db": ra_db(nbs, gt_hd[h]),
             "base_cxcorr": cx_corr(base, gt_hd[h]), "base_db": ra_db(base, gt_hd[h])}
        rows.append(r)
        print(f"  held {h:>4} GHz | NBS cxcorr={r['nbs_cxcorr']:.3f} dB={r['nbs_db']:6.1f} "
              f"| interp cxcorr={r['base_cxcorr']:.3f} dB={r['base_db']:6.1f}")

    print(f"\n  recovered sigma_h err: {(100*(sh_rec-SH).abs()/SH).cpu().numpy().round(0)} %")
    out = os.path.join("output", "gitu_roughness"); os.makedirs(out, exist_ok=True)
    json.dump({"train": TRAIN, "held": HELD, "rows": rows,
               "sigma_h_gt": SH.tolist(), "sigma_h_rec": sh_rec.tolist()},
              open(os.path.join(out, "results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = [r["held"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(h, [r["nbs_db"] for r in rows], "o-", lw=2, label="NBS (recovers dispersion + roughness)")
    ax[0].plot(h, [r["base_db"] for r in rows], "s--", lw=2, label="interp reflectivity attribute (XBandGS-style)")
    ax[0].axvspan(min(TRAIN), max(TRAIN), color="green", alpha=0.07, label="train (smooth regime)")
    ax[0].set_xlabel("held-out carrier (GHz)"); ax[0].set_ylabel("|RA| error (dB)")
    ax[0].set_title("Roughness regime change: |RA| fidelity"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(h, [r["nbs_cxcorr"] for r in rows], "o-", lw=2, label="NBS")
    ax[1].plot(h, [r["base_cxcorr"] for r in rows], "s--", lw=2, label="interp")
    ax[1].set_xlabel("held-out carrier (GHz)"); ax[1].set_ylabel("complex-RA corr"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("complex-RA correlation")
    fig.tight_layout(); fig.savefig(os.path.join(out, "roughness_sweep.png"), dpi=110); plt.close(fig)
    print(f"saved {out}/")


if __name__ == "__main__":
    main()
