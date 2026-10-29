"""UWB single-band per-point fit (experiment, 2026-10-29): the UWB deficit is coarse-scale (trained 0.68 at 39 cm smoothing
vs a 0.98 ceiling), so test the levers the analysis points at: (a) a near guard of 1.0 m (node body, like SFCW's 0.9 m),
(b) per-point six-parameter material with roughness bounds appropriate to lambda = 4.3 cm (sigma_h up to 2 cm instead
of 3DPS's 0.9 mm clamp, physics passed directly). The pose sweep is left to the yaw tool.
    cd 4DPS && CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_fit_uwb [--guard-m 1.0] [--sigma-max 0.02] [--steps 300]
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
from preprocessing import octonet_config as C
from method.freq_material import eps_from_dispersion, ITU_GT
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)

def smooth(x, w):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--points", type=int, default=20000); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--guard-m", type=float, default=0.29); ap.add_argument("--sigma-max", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=0.05); ap.add_argument("--floor", action="store_true", help="learn an additive noise floor on the envelope (GT flattens to a floor past ~9 m)"); ap.add_argument("--device", default="cuda"); ap.add_argument("--tag", default="")
    a = ap.parse_args(argv); dev = torch.device(a.device); torch.manual_seed(0)
    cloud, _, _ = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=a.points)
    sensor = OH.sensor_from_extrinsics(json.load(open(os.path.join(DATA, "calib_S1", "extrinsics.json"))), "uwb")
    z = np.load(os.path.join(DATA, "uwb_target_S1", "target_pool5.npz")); env_gt = z["env"]; rel = z["rel"]; rng = z["range_m"]
    mask = (rng >= a.guard_m) & z["mask"].astype(bool); mt = torch.as_tensor(mask, device=dev); w = torch.as_tensor(np.clip(rel, 0, 1), device=dev, dtype=torch.float64)
    per = z["per_rec_env"]
    n = cloud["xyz"].shape[0]
    er, ei = eps_from_dispersion(torch.tensor(ITU_GT["concrete"], dtype=torch.float64), 7.0)
    lo = torch.tensor([1.0, 0.01, 1e-5, 1e-3, 0.05, 1e-3], device=dev, dtype=torch.float64); hi = torch.tensor([15.0, 5.0, a.sigma_max, 0.3, 0.95, 0.5], device=dev, dtype=torch.float64)
    init = torch.tensor([float(er), float(ei), 1e-4, 1e-2, 0.5, 0.1], device=dev, dtype=torch.float64).clamp(lo * 1.001, hi * 0.999)
    def to_phys(u): return lo + (hi - lo) * torch.sigmoid(u)
    def to_u(p): x = ((p - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4); return torch.log(x / (1 - x))
    u0 = to_u(init)[None, :].expand(n, 6).clone(); u = u0.clone().requires_grad_(True)
    log_floor = torch.tensor(float(np.log(np.percentile(env_gt[mask], 2) + 1e-12)) - 2.0, device=dev, dtype=torch.float64, requires_grad=True)
    g_env = torch.as_tensor(np.log(env_gt + 1e-12), device=dev, dtype=torch.float64)
    def loss_of(env):
        le = torch.log(env + 1e-12); d = (le[mt] - le[mt].median()) - (g_env[mt] - g_env[mt].median()); return (w[mt] * d.abs()).sum() / w[mt].sum()
    def render(uu):
        rf, env = OH.uwb_head(cloud, {"physics": to_phys(uu).float()}, sensor)
        if a.floor:
            # the GT envelope is median-normalised in the loss, so the floor is expressed relative to the render median
            env = env + torch.exp(log_floor) * env[mt].median().detach() / torch.as_tensor(np.median(env_gt[mask]), device=dev)
        return env
    opt = torch.optim.Adam([u] + ([log_floor] if a.floor else []), lr=a.lr); t0 = time.time(); best = (1e9, None, 0)
    with torch.no_grad(): L0 = float(loss_of(render(u)))
    for step in range(1, a.steps + 1):
        opt.zero_grad(); env = render(u); L = loss_of(env); reg = a.l2 * ((u - u0) ** 2).mean(); (L + reg).backward(); opt.step()
        if float(L) < best[0]: best = (float(L), u.detach().clone(), step)
        if step % 100 == 0: print(f"  step {step} loss {float(L):.4f} reg {float(reg):.4f} {time.time()-t0:.0f} s")
    with torch.no_grad():
        env1 = render(best[1]).cpu().numpy(); phys = to_phys(best[1]).cpu().numpy()
    def c(a_, b_, wdw): A = smooth(np.log(a_ + 1e-12), wdw)[mask]; B = smooth(np.log(b_ + 1e-12), wdw)[mask]; return float(np.corrcoef(A - np.median(A), B - np.median(B))[0, 1])
    res = dict(guard_m=a.guard_m, sigma_max=a.sigma_max, floor=(float(torch.exp(log_floor)) if a.floor else None), loss0=L0, loss1=best[0], best_step=best[2],
               corr_vs_pool={str(wd): c(env1, env_gt, wd) for wd in (1, 15, 61)}, corr_vs_each_rec_1bin=[c(env1, e, 1) for e in per],
               ceiling_rec_vs_pool_1bin=float(np.mean([c(e, env_gt, 1) for e in per])), sigma_h_median_mm=float(np.median(phys[:, 2]) * 1e3), sigma_h_p95_mm=float(np.percentile(phys[:, 2], 95) * 1e3),
               eps_r_median=float(np.median(phys[:, 0])), seconds=time.time() - t0)
    print(json.dumps(res))
    out = os.path.join(DATA, "fit_S1_uwb"); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"material{a.tag}.npz"), xyz=cloud["xyz"].cpu().numpy(), physics=phys, env=env1); json.dump(res, open(os.path.join(out, f"fit{a.tag}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
