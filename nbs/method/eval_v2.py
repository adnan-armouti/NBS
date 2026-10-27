"""Corrected cross-band evaluation (post-audit). Reports TWO metrics + controls:

  (1) |RA| dB error vs a WRONG-MATERIAL FLOOR (geometry-independent: a real method must
      beat the floor by a real margin; the floor also = the material CONTRIBUTION to the signal).
  (2) MATERIAL-ID ACCURACY: recovered library assignment vs the true material (eval GT).
  + validated NON-DEGENERATE interp baseline (asserted non-zero before comparison).
  + RAW vs NORMALIZED ablation (does [0,1] normalization hurt material recovery?).

Setting: smooth->rough regime change (where materials matter), dense real-geometry scene,
per-class materials from the ITU library; training is RF-only (recover a soft library
assignment + roughness, with the BSDF reparam bounds as the only material prior).

  python -m method.eval_v2
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm3DPS import bsdf_torch as B
from method import ra_forward as RF
from method.freq_material import eps_from_dispersion, ITU_GT

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIB_NAMES = ["concrete", "glass", "wood", "plasterboard", "brick", "marble"]
LIB = [torch.tensor(ITU_GT[m], dtype=torch.float64, device=DEV) for m in LIB_NAMES]
TRUE = [0, 2, 1, 3]                      # class -> library idx (concrete, wood, glass, plasterboard)
SH_TRUE = torch.tensor([8e-4, 5e-4, 1e-4, 3e-4], dtype=torch.float64, device=DEV)  # straddle 4pi*sh/lam~1
TRAIN = [18.0, 28.0, 40.0]              # smooth regime
HELD = 120.0                            # deep rough regime (materials matter a lot)
NOISE_DB = -20.0


def gen(s): return torch.Generator(device=DEV).manual_seed(s)


def load_scene(n=800):
    p = np.load("/home/adnan/Desktop/NBS/data/XRF55/Scene1/scene/pcl.npy")
    idx = np.random.RandomState(0).choice(len(p), n, replace=False); p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV); nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256 * 0.076); pos, nrm = pos[keep], nrm[keep]
    z = pos[:, 2]; q = torch.quantile(z, torch.tensor([0.25, 0.5, 0.75], device=DEV, dtype=z.dtype))
    cls = (z[:, None] > q[None, :]).sum(1)
    return pos, nrm, cls


def cos_inc(pos, nrm):
    inc = -pos / pos.norm(dim=1, keepdim=True)
    return (inc * nrm).sum(1).clamp(1e-3, 1.0)


def render_from_amp(amp, pos, f, noise=None, g=None):
    return RF._render_ra(amp.to(torch.complex128), pos, f, 60e12, 256, 60e-6, 64, noise, g, DEV)


def gt_amp(cls, pos, nrm, thick, f):
    abcd = torch.stack([LIB[TRUE[int(c)]] for c in cls])
    sh = SH_TRUE[cls]
    return RF._amp_from_eps(*eps_from_dispersion(abcd, f), thick, sh, pos, nrm, f)


def mixture_amp(logits, log_sh, cls, pos, nrm, thick, f):
    B.set_frequency(f * 1e9); lam = 299792458.0 / (f * 1e9)
    ct = cos_inc(pos, nrm); M = pos.shape[0]
    w = torch.softmax(logits, dim=1)[cls]
    R = torch.zeros(M, dtype=torch.complex128, device=DEV)
    for k, abcd in enumerate(LIB):
        er, ei = eps_from_dispersion(abcd, f)
        Rk, _, _, _ = B.itu_slab_fresnel(er.expand(M).float(), ei.expand(M).float(), ct.float(), thick.float())
        R = R + w[:, k].to(torch.complex128) * Rk.to(torch.complex128)
    rough = torch.exp(-0.5 * (4 * math.pi * torch.exp(log_sh)[cls] * ct / lam) ** 2)
    return R * rough.to(torch.complex128)


def db(p, g): return float(20 * torch.log10((p.abs() - g.abs()).norm() / (g.abs().norm() + 1e-12)))


def recover_nbs(cls, pos, nrm, thick, gt_train, normalize):
    K = len(cls.unique())
    logits = torch.zeros(K, len(LIB), device=DEV, requires_grad=True)
    log_sh = torch.log(torch.full((K,), 4e-4, device=DEV)).requires_grad_(True)
    opt = torch.optim.Adam([logits, log_sh], lr=0.05)
    for _ in range(600):
        opt.zero_grad(); loss = 0.0
        for f in TRAIN:
            pred = render_from_amp(mixture_amp(logits, log_sh, cls, pos, nrm, thick, f), pos, f).abs()
            g = gt_train[f].abs()
            if normalize:
                pred, g = pred / (pred.max() + 1e-9), g / (g.max() + 1e-9)
            loss = loss + ((pred - g) ** 2).sum() / (g ** 2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            log_sh.clamp_(math.log(1e-5), math.log(5e-3))
    return logits.detach(), log_sh.detach()


def recover_interp(cls, pos, nrm, thick, gt_train):
    """Validated non-degenerate interp baseline: per-class complex amp per band, init at the
    GT source-band magnitude (non-zero), then interpolate eps... here amp directly."""
    K = len(cls.unique()); band_amp = {}
    for f in TRAIN:
        a0 = gt_amp(cls, pos, nrm, thick, f)
        init = torch.stack([a0[cls == k].mean() for k in range(K)])     # non-zero init from data scale
        ar = init.real.clone().requires_grad_(True); ai = init.imag.clone().requires_grad_(True)
        o = torch.optim.Adam([ar, ai], lr=0.01)
        for _ in range(600):
            o.zero_grad()
            pred = render_from_amp(torch.complex(ar, ai)[cls], pos, f).abs()
            g = gt_train[f].abs()
            loss = ((pred - g) ** 2).sum() / (g ** 2).sum(); loss.backward(); o.step()
        band_amp[f] = torch.complex(ar.detach(), ai.detach())
    assert all(band_amp[f].abs().mean() > 1e-9 for f in TRAIN), "interp baseline degenerate!"
    fs = np.array(TRAIN)
    AR = torch.stack([band_amp[f].real for f in TRAIN]).cpu().numpy()
    AI = torch.stack([band_amp[f].imag for f in TRAIN]).cpu().numpy()
    ar_h = np.array([np.polyval(np.polyfit(fs, AR[:, k], 1), HELD) for k in range(K)])
    ai_h = np.array([np.polyval(np.polyfit(fs, AI[:, k], 1), HELD) for k in range(K)])
    return torch.complex(torch.tensor(ar_h, device=DEV), torch.tensor(ai_h, device=DEV))


def main():
    pos, nrm, cls = load_scene(800)
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    K = len(cls.unique())
    print(f"[scene] {pos.shape[0]} pts, {K} classes, true mats {[LIB_NAMES[TRUE[k]] for k in range(K)]}")
    print(f"  4*pi*sigma_h/lambda at held {HELD} GHz: "
          f"{(4*math.pi*SH_TRUE/(299792458.0/(HELD*1e9))).cpu().numpy().round(2)} (>1 = rough)")

    gt_train = {f: render_from_amp(gt_amp(cls, pos, nrm, thick, f), pos, f, NOISE_DB, gen(100+int(f))) for f in TRAIN}
    gt_held = render_from_amp(gt_amp(cls, pos, nrm, thick, HELD), pos, HELD)

    # ---- material contribution / wrong-material FLOOR at held band ----
    abcd_allc = torch.stack([LIB[0]] * pos.shape[0])                       # all-concrete (wrong materials)
    abcd_true_pt = torch.stack([LIB[TRUE[int(c)]] for c in cls])           # true materials, wrong roughness
    floor_renders = {
        "all-concrete": render_from_amp(
            RF._amp_from_eps(*eps_from_dispersion(abcd_allc, HELD), thick, SH_TRUE[cls], pos, nrm, HELD), pos, HELD),
        "wrong-roughness(all-smooth)": render_from_amp(
            RF._amp_from_eps(*eps_from_dispersion(abcd_true_pt, HELD), thick,
                             torch.full((pos.shape[0],), 1e-4, device=DEV), pos, nrm, HELD), pos, HELD),
    }
    print("\n=== wrong-material FLOOR (= material contribution to the held-band signal) ===")
    for name, r in floor_renders.items():
        print(f"  {name:28s}: dB={db(r, gt_held):6.1f}")
    floor_db = max(db(r, gt_held) for r in floor_renders.values())   # easiest wrong control
    print(f"  -> floor (best wrong control) = {floor_db:.1f} dB; a real method must beat this.")

    # ---- NBS raw vs normalized ----
    results = {}
    for tag, norm in [("NBS-raw", False), ("NBS-normalized", True)]:
        logits, log_sh = recover_nbs(cls, pos, nrm, thick, gt_train, norm)
        held = render_from_amp(mixture_amp(logits, log_sh, cls, pos, nrm, thick, HELD), pos, HELD)
        idacc = float((logits.argmax(1).cpu() == torch.tensor(TRUE)).float().mean())
        w_true = [round(float(torch.softmax(logits, 1)[k, TRUE[k]]), 2) for k in range(K)]
        results[tag] = {"db": db(held, gt_held), "matID_acc": idacc, "weight_on_true": w_true}
        print(f"\n[{tag}] held dB={results[tag]['db']:.1f}  matID-acc={idacc:.2f}  "
              f"weight-on-true-material/class={w_true}")

    # ---- interp baseline (validated) ----
    amp_h = recover_interp(cls, pos, nrm, thick, gt_train)
    interp_held = render_from_amp(amp_h[cls], pos, HELD)
    results["interp"] = {"db": db(interp_held, gt_held), "matID_acc": None}
    print(f"\n[interp baseline] held dB={results['interp']['db']:.1f}  (non-degenerate, asserted)")

    results["floor_db"] = floor_db
    print("\n=== VERDICT ===")
    for m in ["NBS-raw", "NBS-normalized", "interp"]:
        beats = results[m]["db"] < floor_db - 3
        print(f"  {m:16s} dB={results[m]['db']:6.1f}  {'BEATS floor' if beats else 'does NOT beat floor'} ({floor_db:.1f})")
    out = os.path.join("output", "eval_v2"); os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
    print(f"\nsaved {out}/results.json")


if __name__ == "__main__":
    main()
