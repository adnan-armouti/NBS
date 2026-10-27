"""G-ITU GATE (PLAN_v2): does a differentiable ITU-dispersion inverse loop work?

Toy scene of point scatterers with known geometry (the scaffold) and ITU materials.
We render the complex FMCW ADC at several TRAINING bands, then:
  (NBS)  recover the dispersion coeffs (a,b,c,d) per scatterer by back-propagating a
         complex-signal loss, and ANALYTICALLY CONTINUE them to a HELD-OUT band.
  (base) recover per-band complex permittivity independently, then INTERPOLATE eps to
         the held-out band (the field-interpolation analogue of XBandGS et al.).
Compare both to ground truth at the held-out band (complex-ADC correlation + dB error).

PASS = NBS recovers the held-out band's raw signal well (and beats interpolation,
especially when extrapolating). Run in the mmir env:
  python -m method.gitu_gate
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import freq_material as FM

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- scene + radar config ----
MATS = ["concrete", "glass", "wood", "plasterboard", "brick"]
RANGES = torch.tensor([1.4, 2.1, 3.0, 4.2, 5.5], dtype=torch.float64, device=DEV)
GT = torch.tensor([FM.ITU_GT[m] for m in MATS], dtype=torch.float64, device=DEV)  # (M,4)
S = 60e12          # 60 MHz/us chirp slope
K = 256
CHIRP_T = 60e-6

# ---- modulation switch: FMCW (default, unchanged) or SFCW (Vayyar-style stepped sweep) ----
# NBS_MODULATION=SFCW re-runs the same dispersion-recovery gate on the OctoNet/Vayyar modulation:
# each "band" f becomes the CENTER of an SFCW_N-tone sweep of SFCW_BW_GHZ bandwidth, rendered via
# method.sfcw_forward (Hofmann Eq.10 with discrete f_c, no chirp slope term).
MODULATION = os.environ.get("NBS_MODULATION", "FMCW").upper()
SFCW_N = int(os.environ.get("NBS_SFCW_N", "100"))       # freq points (Vayyar = 100)
SFCW_BW_GHZ = float(os.environ.get("NBS_SFCW_BW_GHZ", "4.0"))  # sweep bandwidth (Vayyar = 4 GHz)


def _sfcw_sweep_hz(center_ghz):
    """SFCW_N tone frequencies (Hz) spanning SFCW_BW_GHZ centered on `center_ghz`."""
    lo = center_ghz - SFCW_BW_GHZ / 2.0
    return (lo + np.arange(SFCW_N) * (SFCW_BW_GHZ / (SFCW_N - 1))) * 1e9


def cplx_corr(a, b):
    a = a.flatten(); b = b.flatten()
    a = a - a.mean(); b = b - b.mean()
    num = (a.conj() * b).sum().abs()
    den = torch.sqrt((a.conj() * a).sum().real * (b.conj() * b).sum().real) + 1e-12
    return (num / den).item()


def db_err(pred, gt):
    return (20 * torch.log10((pred - gt).abs().norm() / (gt.abs().norm() + 1e-12))).item()


def render(abcd, f):
    """Complex signal for dispersion coeffs `abcd` at band `f` (GHz). FMCW ADC or SFCW sweep."""
    if MODULATION == "SFCW":
        from method.sfcw_forward import render_sfcw_mono
        return render_sfcw_mono(abcd.to(DEV), RANGES, _sfcw_sweep_hz(f))
    return FM.render_adc(abcd.to(DEV), RANGES, f, S, K, CHIRP_T)


def render_eps(er, ei, f):
    """Complex signal from per-band complex permittivity (er,ei) directly (interpolation baseline).
    Branches on MODULATION exactly like `render`, so the baseline is compared on the same signal."""
    n, kap = FM.permittivity_to_ior(er, ei)
    refl = FM.fresnel_rs(n, kap, 1.0)
    geom = 1.0 / (RANGES ** 2 + 1e-6)
    if MODULATION == "SFCW":
        from method.sfcw_forward import render_sfcw_amp
        return render_sfcw_amp(refl * geom.to(refl.dtype), RANGES, _sfcw_sweep_hz(f))
    t = torch.arange(K, dtype=torch.float64, device=DEV) * (CHIRP_T / K)
    tau = 2 * RANGES / FM.C_LIGHT
    phase = 2 * math.pi * (f * 1e9 * tau)[:, None] + 2 * math.pi * (S * tau)[:, None] * t[None, :]
    return ((refl * geom.to(refl.dtype))[:, None]
            * torch.exp(1j * phase.to(torch.complex128))).sum(0).to(torch.complex64)


def run(train_bands, heldout, n_iter=1500, lr=0.05):
    gt_train = {f: render(GT, f) for f in train_bands}
    gt_held = render(GT, heldout)

    # ---------- NBS: recover dispersion (a,b,c,d), continue to held-out ----------
    init = GT.clone(); init[:, 0] *= 1.6; init[:, 2] *= 0.4; init[:, 3] *= 1.3   # wrong start
    p = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = sum(((render(p, f) - gt_train[f]).abs() ** 2).mean() for f in train_bands)
        loss.backward(); opt.step()
        with torch.no_grad():                       # keep physical
            p.clamp_(min=torch.tensor([1.0, -1.0, 1e-5, 0.0], device=DEV))
    with torch.no_grad():
        nbs_held = render(p.detach(), heldout)
    nbs = {"corr": cplx_corr(nbs_held, gt_held), "db": db_err(nbs_held, gt_held)}

    # ---------- baseline: per-band complex eps, then interpolate eps -> held-out ----------
    # recover (eps_real, eps_imag) per scatterer independently at each train band
    band_eps = {}
    for f in train_bands:
        er = (GT[:, 0] * 1.6).clone().requires_grad_(True)
        ei = (17.98 * GT[:, 2] * f ** (GT[:, 3] - 1) * 0.4).clone().requires_grad_(True)
        o = torch.optim.Adam([er, ei], lr=lr)
        for _ in range(n_iter):
            o.zero_grad()
            s = render_eps(er, ei, f)
            loss = ((s - gt_train[f]).abs() ** 2).mean()
            loss.backward(); o.step()
        band_eps[f] = (er.detach(), ei.detach())
    # linear interpolate/extrapolate eps across train bands to held-out
    fs = torch.tensor(train_bands, dtype=torch.float64, device=DEV)
    ers = torch.stack([band_eps[f][0] for f in train_bands])    # (B,M)
    eis = torch.stack([band_eps[f][1] for f in train_bands])
    def interp(vals):
        out = []
        for m in range(len(RANGES)):
            out.append(np.interp(heldout, fs.cpu().numpy(), vals[:, m].cpu().numpy(),
                                 left=None, right=None) if (fs.min() <= heldout <= fs.max())
                       else _lin_extrap(fs.cpu().numpy(), vals[:, m].cpu().numpy(), heldout))
        return torch.tensor(out, dtype=torch.float64, device=DEV)
    er_h, ei_h = interp(ers), interp(eis)
    base_held = render_eps(er_h, ei_h, heldout)
    base = {"corr": cplx_corr(base_held, gt_held), "db": db_err(base_held, gt_held)}
    return nbs, base


def _lin_extrap(x, y, xq):
    # linear fit extrapolation
    A = np.polyfit(x, y, 1)
    return float(np.polyval(A, xq))


def band_gap_sweep(train=(10.0, 24.0, 40.0), held_bands=(30, 45, 55, 65, 77, 90, 100)):
    """Signal-fidelity vs band gap: NBS analytic dispersion vs interp baseline."""
    rows = []
    for h in held_bands:
        nbs, base = run(list(train), float(h), n_iter=1200)
        rows.append((h, nbs["db"], base["db"]))
        print(f"  held {h:>4} GHz: NBS dB={nbs['db']:6.1f} | interp dB={base['db']:6.1f}")
    return train, rows


def main():
    scenarios = [
        ("interpolation", [10.0, 24.0, 77.0], 40.0),
        ("extrapolation", [10.0, 24.0, 40.0], 77.0),
    ]
    results = {}
    for name, train, held in scenarios:
        nbs, base = run(train, held)
        results[name] = {"train_bands": train, "heldout": held, "nbs": nbs, "interp_baseline": base}
        print(f"[{name}] held-out {held} GHz from {train}")
        print(f"   NBS (analytic dispersion):   complex-corr={nbs['corr']:.4f}  dB-err={nbs['db']:.1f}")
        print(f"   interp baseline (per-band):  complex-corr={base['corr']:.4f}  dB-err={base['db']:.1f}")

    print("\n[band-gap sweep] train on {10,24,40} GHz, vary held-out band:")
    train, rows = band_gap_sweep()
    results["band_gap_sweep"] = {"train_bands": list(train),
                                 "rows": [{"held_ghz": h, "nbs_db": n, "interp_db": b} for h, n, b in rows]}

    out = os.path.join("output", "gitu_gate"); os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = [r[0] for r in rows]; nb = [r[1] for r in rows]; bb = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(h, nb, "o-", label="NBS (analytic ITU dispersion)", lw=2)
    ax.plot(h, bb, "s--", label="per-band + interpolation (field-interp analogue)", lw=2)
    for f in train:
        ax.axvline(f, color="gray", ls=":", alpha=0.5)
    ax.axvspan(min(train), max(train), color="green", alpha=0.06, label="train-band span (interp region)")
    ax.set_xlabel("held-out carrier (GHz)"); ax.set_ylabel("synth-signal error (dB, lower=better)")
    ax.set_title("G-ITU: raw-signal fidelity vs band gap"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "band_gap_sweep.png"), dpi=110); plt.close(fig)

    ok = all(results[s]["nbs"]["corr"] > 0.9 for s in ("interpolation", "extrapolation"))
    print(f"\nG-ITU GATE: {'PASS' if ok else 'CHECK'} (NBS complex-corr > 0.9 held-out: {ok})")
    print(f"saved {out}/band_gap_sweep.png + results.json")


if __name__ == "__main__":
    main()
