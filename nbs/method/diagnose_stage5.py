"""Re-audit Stage 5 (and the roughness test pattern) with the wrong-material control:
does a WRONG-material render already achieve NBS's dB? is the interp baseline degenerate?"""
import os, sys, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import ra_forward as RF
from method.freq_material import eps_from_dispersion, ITU_GT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIB = ["wood", "plasterboard", "concrete", "glass"]
SH = torch.tensor([8e-4, 5e-4, 3e-4, 1e-4], dtype=torch.float64, device=DEV)


def db(p, g): return float(20*torch.log10((p.abs()-g.abs()).norm()/(g.abs().norm()+1e-12)))
def corr(p, g):
    p, g = p.abs().flatten(), g.abs().flatten(); p, g = p-p.mean(), g-g.mean()
    return float((p*g).sum()/(p.norm()*g.norm()+1e-12))


def main():
    p = np.load("/home/adnan/Desktop/NBS/data/XRF55/Scene1/scene/pcl.npy")
    idx = np.random.RandomState(0).choice(len(p), 2000, replace=False)
    p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV); nrm = nrm/nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256*0.076); pos, nrm = pos[keep], nrm[keep]
    z = pos[:, 2]; q = torch.quantile(z, torch.tensor([0.25, 0.5, 0.75], device=DEV, dtype=z.dtype))
    cls = (z[:, None] > q[None, :]).sum(1)
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    HELD = 90.0

    def amp(class_mat, sh_cls, f):
        abcd = torch.stack([torch.tensor(ITU_GT[class_mat[int(c)]], dtype=torch.float64, device=DEV) for c in cls])
        sh = sh_cls[cls]
        return RF._amp_from_eps(*eps_from_dispersion(abcd, f), thick, sh, pos, nrm, f)
    def render(a, f): return RF._render_ra(a, pos, f, 60e12, 256, 60e-6, 64, None, None, DEV)

    gt = render(amp(LIB, SH, HELD), HELD)

    print("=== Stage 5 wrong-material control at held 90 GHz ===")
    print(f"  GT |RA| mean={gt.abs().mean():.3e}")
    # all-concrete (wrong)
    allc = render(amp(["concrete"]*4, SH, HELD), HELD)
    # shuffled materials
    shuf = render(amp(["glass", "concrete", "wood", "plasterboard"], SH, HELD), HELD)
    # wrong roughness (all smooth)
    smooth = render(amp(LIB, torch.full((4,), 1e-4, device=DEV), HELD), HELD)
    print(f"  all-concrete   : corr={corr(allc, gt):.4f} dB={db(allc, gt):6.1f}")
    print(f"  shuffled mats  : corr={corr(shuf, gt):.4f} dB={db(shuf, gt):6.1f}")
    print(f"  all-smooth sigma: corr={corr(smooth, gt):.4f} dB={db(smooth, gt):6.1f}")
    print("  (Stage 5 claimed NBS dB=-37.5, interp dB=-5.9. If wrong materials already")
    print("   reach ~-37 dB, the Stage-5 number is geometry-dominated too.)")


if __name__ == "__main__":
    main()
