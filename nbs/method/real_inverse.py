"""Diagnose + close the sim-to-real RA gap, in stages, so we know WHAT the gap is:
  (0) default forward render vs real RA                      -> baseline corr
  (1) + CALIBRATION (range scale/shift, azimuth shift/flip, gain): the unknown chirp config
      and angle convention are free scalars; this is cheap and must be done before anything else
  (2) + MATERIAL inverse rendering (per-class library + roughness)
Reports dB-correlation after each stage, so we attribute the gap to calibration vs geometry/physics.

  python -m method.real_inverse --scene Scene1 --index 0
"""
import os, sys, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm3DPS import bsdf_torch as B
from method import ra_forward as RF
from method.freq_material import eps_from_dispersion, ITU_GT
from preprocessing import xrf55_io as IO, xrf55_config as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S60 = 299792458.0 / (2 * 0.076) / 60e-6
CHIRP_T = 60e-6
LIB = ["concrete", "wood", "plasterboard", "glass", "brick", "marble"]


def dbimg(x):
    x = x.abs() if torch.is_complex(x) else x
    return torch.clamp(20 * torch.log10(x / (x.max() + 1e-9) + 1e-4), -40, 0)


def corr(a, b):
    a, b = a.flatten(), b.flatten(); a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


def warp(img, rs, rsh, ash, flip, az_s=1.0):
    """Affine resample of a (R,A) image: range scale rs + shift rsh, azimuth scale az_s + shift ash (+flip)."""
    R, A = img.shape
    rr = torch.linspace(-1, 1, R, device=img.device)
    aa = torch.linspace(-1, 1, A, device=img.device)
    gr = rr[:, None] * rs + rsh
    ga = (aa[None, :] * (-az_s if flip else az_s)) + ash
    grid = torch.stack([ga.expand(R, A), gr.expand(R, A)], dim=-1)[None]   # (1,R,A,2): (x=az,y=range)
    return F.grid_sample(img[None, None], grid, align_corners=True, padding_mode="border")[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--npts", type=int, default=4000)
    a = ap.parse_args()
    mmw = IO.list_mmwave(a.scene)[a.index]
    real = torch.tensor(np.median(IO.load_mmwave(mmw)["ra"], axis=0), device=DEV).float()   # (256,64)
    real_db = dbimg(real)

    p = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy"))
    idx = np.random.RandomState(0).choice(len(p), min(a.npts, len(p)), replace=False); p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV); nrm = nrm/nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256*0.076); pos, nrm = pos[keep], nrm[keep]
    z = pos[:, 2]; q = torch.quantile(z, torch.tensor([0.33, 0.66], device=DEV, dtype=z.dtype))
    cls = (z[:, None] > q[None, :]).sum(1); K = 3
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)

    def render(abcd, sh):
        return RF.render_ra_disp(abcd, thick, sh, pos, nrm, 60.0, S=S60, K=256, chirp_T=CHIRP_T, n_az=64).abs().T.float()

    abcd0 = torch.stack([torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=DEV)] * pos.shape[0])
    sh0 = torch.full((pos.shape[0],), 3e-4, dtype=torch.float64, device=DEV)
    sim0 = render(abcd0, sh0)
    print(f"[0] default render vs real: corr={corr(dbimg(sim0), real_db).item():.3f}")

    # ---- (1) calibration: optimize warp (range+az scale/shift) + flip on dB images ----
    best = (-1, None)
    for flip in [False, True]:
        rs = torch.tensor(1.0, device=DEV, requires_grad=True)
        rsh = torch.tensor(0.0, device=DEV, requires_grad=True)
        ash = torch.tensor(0.0, device=DEV, requires_grad=True)
        azs = torch.tensor(1.0, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([rs, rsh, ash, azs], lr=0.02)
        sim_db = dbimg(sim0)
        for _ in range(600):
            opt.zero_grad()
            w = warp(sim_db, rs, rsh, ash, flip, azs)
            loss = -corr(w, real_db); loss.backward(); opt.step()
        c = corr(warp(dbimg(sim0), rs.detach(), rsh.detach(), ash.detach(), flip, azs.detach()), real_db).item()
        if c > best[0]:
            best = (c, (float(rs), float(rsh), float(ash), flip, float(azs)))
    print(f"[1] + calibration (range+az warp+flip): corr={best[0]:.3f}  "
          f"params(rs,rsh,ash,flip,azs)={tuple(round(x,3) if isinstance(x,float) else x for x in best[1])}")

    # ---- (2) + material inverse rendering with calibration fixed at best ----
    rs, rsh, ash, flip, azs = best[1]
    logits = torch.zeros(K, len(LIB), device=DEV, requires_grad=True)
    log_sh = torch.log(torch.full((K,), 4e-4, device=DEV)).requires_grad_(True)
    opt = torch.optim.Adam([logits, log_sh], lr=0.05)
    LIBt = [torch.tensor(ITU_GT[m], dtype=torch.float64, device=DEV) for m in LIB]
    for _ in range(300):
        opt.zero_grad()
        Bset = B.set_frequency(60e9)
        ct = (-pos / pos.norm(dim=1, keepdim=True) * nrm).sum(1).clamp(1e-3, 1.0)
        w = torch.softmax(logits, 1)[cls]
        Rc = torch.zeros(pos.shape[0], dtype=torch.complex128, device=DEV)
        for k, ab in enumerate(LIBt):
            er, ei = eps_from_dispersion(ab, 60.0)
            Rk, _, _, _ = B.itu_slab_fresnel(er.expand(pos.shape[0]).float(), ei.expand(pos.shape[0]).float(), ct.float(), thick.float())
            Rc = Rc + w[:, k].to(torch.complex128) * Rk.to(torch.complex128)
        rough = torch.exp(-0.5 * (4*math.pi*torch.exp(log_sh)[cls]*ct/(299792458.0/60e9))**2)
        amp = Rc * rough.to(torch.complex128)
        sim = RF._render_ra(amp, pos, 60.0, S60, 256, CHIRP_T, 64, None, None, DEV).abs().T.float()
        loss = -corr(warp(dbimg(sim), torch.tensor(rs, device=DEV), torch.tensor(rsh, device=DEV),
                          torch.tensor(ash, device=DEV), flip, torch.tensor(azs, device=DEV)), real_db)
        loss.backward(); opt.step()
        with torch.no_grad(): log_sh.clamp_(math.log(1e-5), math.log(5e-3))
    print(f"[2] + material inverse render: corr={(-loss).item():.3f}")
    print("\nATTRIBUTION: stage1-stage0 = calibration gap; remaining (1 - stage1) = geometry/physics gap.")


if __name__ == "__main__":
    main()
