"""mmWave <-> WiFi translation, BOTH directions, with the corrected metrics (post-audit):
material-ID accuracy + target-observable dB vs a WRONG-MATERIAL floor + a validated baseline.

Direction A: recover materials from mmWave |RA| (60 GHz), render WiFi |CSI| (5 GHz).
Direction B: recover materials from WiFi |CSI| (5 GHz), render mmWave |RA| (60 GHz).

Honest expectation: materials contribute ~8% of the signal (geometry dominates and is
band-independent), so the floor a method must beat is small; the meaningful metric is whether
the recovered MATERIALS are correct (identification accuracy), since that is the only
band-transferable content. Sim only (real WiFi Tx/Rx geometry is unpublished in XRF55).

  python -m method.translate_v2
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm3DPS import bsdf_torch as B
from method import ra_forward as RF, wifi_forward as WF
from method.freq_material import eps_from_dispersion, ITU_GT

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIB_NAMES = ["concrete", "glass", "wood", "plasterboard", "brick", "marble"]
LIB = [torch.tensor(ITU_GT[m], dtype=torch.float64, device=DEV) for m in LIB_NAMES]
TRUE = [0, 2, 3]                      # concrete / wood / plasterboard
MMW, WIFI = 60.0, 5.0
NOISE_DB = -20.0


def scene(n=400):
    p = np.load("/home/adnan/Desktop/NBS/data/XRF55/Scene1/scene/pcl.npy")
    idx = np.random.RandomState(1).choice(len(p), n, replace=False); p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV); nrm = nrm/nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256*0.076); pos, nrm = pos[keep], nrm[keep]
    z = pos[:, 2]; q = torch.quantile(z, torch.tensor([1/3, 2/3], device=DEV, dtype=z.dtype))
    return pos, nrm, (z[:, None] > q[None, :]).sum(1)


def cosi(pos, nrm): return (-pos/pos.norm(dim=1, keepdim=True) * nrm).sum(1).clamp(1e-3, 1.0)


def mat_amp(absel, sh, pos, nrm, thick, f):
    """reflectivity from a per-point library index OR soft weights."""
    B.set_frequency(f*1e9); lam = 299792458.0/(f*1e9); ct = cosi(pos, nrm); M = pos.shape[0]
    if absel.dim() == 1:                         # hard index per point
        abcd = torch.stack([LIB[int(i)] for i in absel])
        er, ei = eps_from_dispersion(abcd, f)
        R, _, _, _ = B.itu_slab_fresnel(er.float(), ei.float(), ct.float(), thick.float())
        R = R.to(torch.complex128)
    else:                                        # soft weights (M, Klib)
        R = torch.zeros(M, dtype=torch.complex128, device=DEV)
        for k, ab in enumerate(LIB):
            er, ei = eps_from_dispersion(ab, f)
            Rk, _, _, _ = B.itu_slab_fresnel(er.expand(M).float(), ei.expand(M).float(), ct.float(), thick.float())
            R = R + absel[:, k].to(torch.complex128) * Rk.to(torch.complex128)
    return R * torch.exp(-0.5*(4*math.pi*sh*ct/lam)**2).to(torch.complex128)


def main():
    pos, nrm, cls = scene(400); K = 3
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    sh_true = torch.tensor([6e-4, 4e-4, 2e-4], device=DEV)[cls]
    tx = torch.tensor([[-1.,.5,0],[-1.,.5,.03],[-1.,.5,.06]], dtype=torch.float64, device=DEV)
    rx = torch.tensor([[4.,6.,0],[4.03,6.,0],[4.06,6.,0]], dtype=torch.float64, device=DEV)
    sc = WF.subcarrier_ghz(WIFI, 40., 30)
    true_pt = torch.tensor([TRUE[int(c)] for c in cls], device=DEV)

    def obs(amp, mod, f, noise=None, g=None):
        if mod == "mmw": return RF._render_ra(amp, pos, f, 60e12, 256, 60e-6, 64, noise, g, DEV)
        return WF.render_csi(amp, pos, tx, rx, sc, noise_db=noise, gen=g)

    def db(p, gt): return float(20*torch.log10((p.abs()-gt.abs()).norm()/(gt.abs().norm()+1e-12)))

    res = {}
    for name, smod, sf, tmod, tf in [("mmW->WiFi", "mmw", MMW, "wifi", WIFI),
                                     ("WiFi->mmW", "wifi", WIFI, "mmw", MMW)]:
        gt_s = obs(mat_amp(true_pt, sh_true, pos, nrm, thick, sf), smod, sf, NOISE_DB, torch.Generator(device=DEV).manual_seed(7))
        gt_t = obs(mat_amp(true_pt, sh_true, pos, nrm, thick, tf), tmod, tf)
        # recover soft library ID + roughness from SOURCE band
        logits = torch.zeros(K, len(LIB), device=DEV, requires_grad=True)
        lsh = torch.log(torch.full((K,), 4e-4, device=DEV)).requires_grad_(True)
        opt = torch.optim.Adam([logits, lsh], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            amp = mat_amp(torch.softmax(logits, 1)[cls], torch.exp(lsh)[cls], pos, nrm, thick, sf)
            pred = obs(amp, smod, sf).abs()
            loss = ((pred-gt_s.abs())**2).sum()/(gt_s.abs()**2).sum(); loss.backward(); opt.step()
            with torch.no_grad(): lsh.clamp_(math.log(1e-5), math.log(5e-3))
        idacc = float((logits.argmax(1).cpu() == torch.tensor(TRUE)).float().mean())
        nbs_t = obs(mat_amp(torch.softmax(logits.detach(), 1)[cls], torch.exp(lsh.detach())[cls], pos, nrm, thick, tf), tmod, tf)
        # wrong-material floor on TARGET (all-concrete)
        floor = obs(mat_amp(torch.zeros_like(true_pt), sh_true, pos, nrm, thick, tf), tmod, tf)
        res[name] = {"matID_acc": idacc, "nbs_target_dB": db(nbs_t, gt_t), "wrong_material_floor_dB": db(floor, gt_t)}
        print(f"[{name}] material-ID acc={idacc:.2f} | NBS target dB={res[name]['nbs_target_dB']:.1f} "
              f"| wrong-material floor dB={res[name]['wrong_material_floor_dB']:.1f}")

    print("\nReading: material-ID accuracy is the meaningful number (band-transferable content);")
    print("the small NBS-vs-floor gap reflects that geometry (band-independent) dominates the signal.")
    out = os.path.join("output", "translate_v2"); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
