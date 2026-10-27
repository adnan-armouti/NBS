"""Adversarial self-audit of band_translate: is the high correlation real, or is it
geometry-dominated (so ANY materials score high), and is the copy baseline degenerate?"""
import os, sys, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF, wifi_forward as WF
from method import band_translate as BT
from method.freq_material import eps_from_dispersion

DEV = BT.DEV


def corr(p, g):
    p, g = p.abs().flatten(), g.abs().flatten(); p, g = p - p.mean(), g - g.mean()
    return float((p * g).sum() / (p.norm() * g.norm() + 1e-12))


def main():
    pos, nrm, cls = BT.scene(300)
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    tx = torch.tensor([[-1.0, 0.5, 0.0], [-1.0, 0.5, 0.03], [-1.0, 0.5, 0.06]], dtype=torch.float64, device=DEV)
    rx = torch.tensor([[4.0, 6.0, 0.0], [4.03, 6.0, 0.0], [4.06, 6.0, 0.0]], dtype=torch.float64, device=DEV)
    sc = WF.subcarrier_ghz(5.0, 40.0, 30)

    def amp(cls_use, f):
        abcd = torch.stack([BT.LIB[BT.TRUE_CLASS_MAT[int(c)]] for c in cls_use])
        sh = torch.tensor([[8e-4, 4e-4, 2e-4][int(c)] for c in cls_use], dtype=torch.float64, device=DEV)
        return RF._amp_from_eps(*eps_from_dispersion(abcd, f), thick, sh, pos, nrm, f)

    # GT mmWave (60 GHz) target
    gt = RF._render_ra(amp(cls, 60.0), pos, 60.0, 60e12, 256, 60e-6, 64, None, None, DEV)

    print("=== TEST B: is corr geometry-dominated? (render target with WRONG materials) ===")
    # all-class-0 material
    wrong_const = RF._render_ra(amp(torch.zeros_like(cls), 60.0), pos, 60.0, 60e12, 256, 60e-6, 64, None, None, DEV)
    # shuffled class labels
    perm = cls[torch.randperm(len(cls), generator=torch.Generator().manual_seed(3)).to(DEV)]
    wrong_shuf = RF._render_ra(amp(perm, 60.0), pos, 60.0, 60e12, 256, 60e-6, 64, None, None, DEV)
    # random complex per-point amplitude (materials totally wrong, SAME geometry)
    g = torch.Generator(device=DEV).manual_seed(5)
    rand_amp = (torch.rand(len(pos), generator=g, dtype=torch.float64, device=DEV)
                * torch.exp(1j * 2 * math.pi * torch.rand(len(pos), generator=g, dtype=torch.float64, device=DEV))).to(torch.complex128)
    wrong_rand = RF._render_ra(rand_amp, pos, 60.0, 60e12, 256, 60e-6, 64, None, None, DEV)
    def db(p, g): return float(20*torch.log10((p.abs()-g.abs()).norm()/(g.abs().norm()+1e-12)))
    print(f"  all-concrete    : corr={corr(wrong_const, gt):.4f}  dB={db(wrong_const, gt):6.1f}")
    print(f"  shuffled-class  : corr={corr(wrong_shuf, gt):.4f}  dB={db(wrong_shuf, gt):6.1f}")
    print(f"  RANDOM amp      : corr={corr(wrong_rand, gt):.4f}  dB={db(wrong_rand, gt):6.1f}")
    print("  -> corr ~0.9+ => geometry-dominated. Compare wrong-material dB to NBS's claimed dB:")
    print("     does a WRONG-material render already achieve a 'good' dB? if so dB is weak too.")

    print("\n=== TEST C: is the copy baseline degenerate? ===")
    # replicate the copy baseline for wifi->mmwave: fit per-class complex amp to WiFi gt_src, render mmwave
    gt_src = WF.render_csi(amp(cls, 5.0), pos, tx, rx, sc, noise_db=-20.0,
                           gen=torch.Generator(device=DEV).manual_seed(7))
    K = len(cls.unique())
    ar = torch.zeros(K, device=DEV, requires_grad=True); ai = torch.zeros(K, device=DEV, requires_grad=True)
    o = torch.optim.Adam([ar, ai], lr=0.02)
    for _ in range(500):
        o.zero_grad()
        a = torch.complex(ar, ai)[cls]
        pred = WF.render_csi(a.to(torch.complex128), pos, tx, rx, sc)
        loss = ((pred.abs() - gt_src.abs()) ** 2).sum() / (gt_src.abs() ** 2).sum()
        loss.backward(); o.step()
    copy_amp = torch.complex(ar.detach(), ai.detach())
    copy_tgt = RF._render_ra(copy_amp[cls].to(torch.complex128), pos, 60.0, 60e12, 256, 60e-6, 64, None, None, DEV)
    print(f"  recovered copy amp per class: {copy_amp.abs().cpu().numpy().round(3)}")
    print(f"  copy_tgt: any NaN={bool(torch.isnan(copy_tgt).any())}, "
          f"|.| mean={copy_tgt.abs().mean():.3e}, std={copy_tgt.abs().std():.3e}")
    print(f"  corr(GT, copy_tgt) = {corr(copy_tgt, gt):.4f}")
    print("  -> if copy amp ~0 or copy_tgt ~constant, the baseline is broken (unfairly favors NBS).")

    print("\n=== TEST D: a geometry-controlled fidelity metric ===")
    # correlate the per-bin |.| AFTER dividing out an all-concrete render (removes shared geometry envelope)
    def detrended(x):
        return (x.abs() / (wrong_const.abs() + 1e-6)).flatten()
    dg = detrended(gt) - detrended(gt).mean()
    dr = detrended(wrong_rand) - detrended(wrong_rand).mean()
    print(f"  detrended corr(GT, RANDOM amp) = {float((dg*dr).sum()/(dg.norm()*dr.norm()+1e-12)):.4f}")
    print("  (a metric where random materials should score ~0 if it actually measures materials)")


if __name__ == "__main__":
    main()
