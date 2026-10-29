"""node_1 Vayyar mount yaw/pitch sweep through the T23 SFCW head (3DPS rasterizer: Jones BSDF, datasheet element
pattern, hemisphere weight), replacing preprocessing/octonet_mount_yaw.py (deprecated point-sum model, angle-blind).

At each candidate boresight (radar position fixed at the node_1 rigid body) the calibrated scaffold is rendered with
uniform ITU concrete and scored against the pooled static target: (i) sfcw_static_loss (logmag, xcorr-weighted, bins
< 0.9 m masked), (ii) pair-mean log-profile correlation, (iii) per-pair image correlation. Fine yaw sweep +-60 deg
step 2, full circle step 10, pitch -10..+10 at the best yaw, and a short per-point material refit at the 5 best yaws
(does the optimum move?). Writes $NBS_DATA_OUT/calib_S1/mount_yaw_v2.json, viz/render_vs_gt/mount_yaw_sweep_v2.png.
Does NOT touch extrinsics.json.   Run on the node: cd nbs && CUDA_VISIBLE_DEVICES=5 python -m preprocessing.octonet_mount_yaw_v2
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np, torch
from preprocessing import octonet_config as C
from method import sfcw_target as ST
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs

DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
DEV = "cuda"
N_POINTS = 30000
FINE = np.arange(-60, 61, 2.0)
COARSE = np.arange(-180, 180, 10.0)
PITCHES = (-10.0, -5.0, 0.0, 5.0, 10.0)
REFIT_STEPS = 60


def db(x): return 20 * np.log10(np.abs(x) + 1e-300)


def rot_z(v, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def tilt(v, pitch_deg):
    h = np.array([v[0], v[1], 0.0]); h /= np.linalg.norm(h); a = math.radians(pitch_deg)
    return math.cos(a) * h + math.sin(a) * np.array([0.0, 0.0, 1.0])


def load_target(device):
    z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz"))
    t = {k: torch.as_tensor(z[k]) for k in z.files}
    t["rp_mean"] = t["rp_mean"].to(torch.complex128).to(device); t["mask"] = t["mask"].bool().to(device); t["mask"][:24] = False
    for k in ("rel", "xcorr", "snr_db"):
        if k in t: t[k] = t[k].double().to(device)
    return t


class Scorer:
    def __init__(self, cloud, material, sensor, target):
        self.cloud, self.material, self.sensor0, self.t = cloud, material, sensor, target
        self.mask = target["mask"].cpu().numpy()
        G = db(target["rp_mean"].cpu().numpy())[:, self.mask]
        self.G_img = G - np.median(G); self.G_prof = G.mean(0) - np.median(G.mean(0))

    def sensor_at(self, dyaw, pitch=0.0):
        b = tilt(rot_z(np.asarray(self.sensor0["boresight"], float), dyaw), pitch)
        return dict(self.sensor0, boresight=b)

    def score(self, dyaw, pitch=0.0, material=None):
        sens = self.sensor_at(dyaw, pitch)
        with torch.no_grad():
            rp = OH.sfcw_head(self.cloud, material or self.material, sens)
            loss = ST.sfcw_static_loss(rp, self.t, kind="logmag", weight="xcorr")
            loss = float(loss[0] if isinstance(loss, (tuple, list)) else loss)
        R = db(rp.cpu().numpy())[:, self.mask]
        prof = R.mean(0) - np.median(R.mean(0)); img = R - np.median(R)
        OH._SFCW_RAST.clear()                                               # one Rasterizer per pose; free it
        return dict(dyaw=float(dyaw), pitch=float(pitch), loss=loss,
                    corr_profile=float(np.corrcoef(self.G_prof, prof)[0, 1]),
                    corr_image=float(np.corrcoef(self.G_img.ravel(), img.ravel())[0, 1]))

    def refit(self, dyaw, steps=REFIT_STEPS, lr=0.02):
        """Short per-point six-parameter fit at a fixed pose; returns the loss after the fit."""
        from mm3DPS.rasterizer import inverse_reparameterize_torch
        from method.freq_material import eps_from_dispersion
        n = self.cloud["xyz"].shape[0]
        er, ei = eps_from_dispersion(self.material["abcd"][:1].double(), OH.SFCW_FC_GHZ)
        phys0 = np.tile([float(er), float(ei), OH.SFCW_DEFAULT_ROUGHNESS["sigma_h"], OH.SFCW_DEFAULT_ROUGHNESS["l_c"],
                         OH.SFCW_DEFAULT_ROUGHNESS["tau"], OH.SFCW_DEFAULT_ROUGHNESS["thickness"]], (n, 1))
        raw0 = torch.as_tensor(inverse_reparameterize_torch(phys0), dtype=torch.float32, device=self.cloud["xyz"].device)
        raw6 = raw0.clone().requires_grad_(True); opt = torch.optim.Adam([raw6], lr=lr)
        sens = self.sensor_at(dyaw)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            rp = OH.sfcw_head(self.cloud, {"raw6": raw6}, sens)
            l = ST.sfcw_static_loss(rp, self.t, kind="logmag", weight="xcorr"); l = l[0] if isinstance(l, (tuple, list)) else l
            (l + 0.1 * ((raw6 - raw0) ** 2).mean()).backward(); opt.step()
        with torch.no_grad():
            rp = OH.sfcw_head(self.cloud, {"raw6": raw6}, sens)
            l = ST.sfcw_static_loss(rp, self.t, kind="logmag", weight="xcorr"); l = float(l[0] if isinstance(l, (tuple, list)) else l)
        OH._SFCW_RAST.clear()
        return l


def prominence(xs, ys, x0, exclude=20.0):
    xs = np.asarray(xs); ys = np.asarray(ys); away = np.abs(xs - x0) > exclude
    if away.sum() < 3: return float("nan")
    return float((ys[away].mean() - ys[np.argmin(np.abs(xs - x0))]) / (ys[away].std() + 1e-12))


def main():
    t0 = time.time()
    cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=DEV, n_points=N_POINTS)
    target = load_target(DEV); sc = Scorer(cloud, material, sensor, target)
    cam_yaw = math.degrees(math.atan2(sensor["boresight"][1], sensor["boresight"][0]))
    print(f"camera yaw (prior) {cam_yaw:.1f} deg; {N_POINTS} points; head = 3DPS render_factorized")
    fine = [sc.score(d) for d in FINE]
    for r in fine[::5]: print(f"  dyaw {r['dyaw']:+5.0f}: loss {r['loss']:.4f} corr_prof {r['corr_profile']:+.3f} corr_img {r['corr_image']:+.3f}")
    L = np.array([r["loss"] for r in fine]); CP = np.array([r["corr_profile"] for r in fine]); CI = np.array([r["corr_image"] for r in fine])
    i_l, i_p, i_i = int(np.argmin(L)), int(np.argmax(CP)), int(np.argmax(CI))
    best = dict(loss=dict(dyaw=float(FINE[i_l]), value=float(L[i_l]), z=prominence(FINE, L, FINE[i_l])),
                corr_profile=dict(dyaw=float(FINE[i_p]), value=float(CP[i_p]), z=prominence(FINE, -CP, FINE[i_p])),
                corr_image=dict(dyaw=float(FINE[i_i]), value=float(CI[i_i]), z=prominence(FINE, -CI, FINE[i_i])))
    print("fine sweep best:", json.dumps(best))
    coarse = [sc.score(d) for d in COARSE]
    Lc = np.array([r["loss"] for r in coarse]); best["full_circle"] = dict(dyaw=float(COARSE[int(np.argmin(Lc))]), value=float(Lc.min()), z=prominence(COARSE, Lc, COARSE[int(np.argmin(Lc))], exclude=30.0))
    print("full circle best:", best["full_circle"])
    pitch = [sc.score(FINE[i_l], p) for p in PITCHES]
    print("pitch at best yaw:", [(r["pitch"], round(r["loss"], 4)) for r in pitch])
    # refit pass at the 5 best yaws (by loss)
    top = [float(FINE[k]) for k in np.argsort(L)[:5]]
    if 0.0 not in top: top.append(0.0)
    refit = {}
    for d in sorted(top):
        refit[str(d)] = sc.refit(d); print(f"  refit {REFIT_STEPS} steps at dyaw {d:+.0f}: loss {refit[str(d)]:.4f}")
    rb = min(refit, key=refit.get)
    agree = max(abs(best["loss"]["dyaw"] - best["corr_profile"]["dyaw"]), abs(best["loss"]["dyaw"] - best["corr_image"]["dyaw"])) <= 5.0
    measured = agree and best["loss"]["z"] > 3.0 and best["full_circle"]["z"] > 3.0 and abs(best["full_circle"]["dyaw"] - best["loss"]["dyaw"]) <= 10.0
    verdict = (f"MEASURED: yaw offset {best['loss']['dyaw']:+.0f} deg from the camera yaw (loss z {best['loss']['z']:.1f}, metrics agree within 5 deg)" if measured
               else f"NOT MEASURED to the 3-sigma standard: best loss at {best['loss']['dyaw']:+.0f} deg (z {best['loss']['z']:.1f}), profile-corr at {best['corr_profile']['dyaw']:+.0f}, image-corr at {best['corr_image']['dyaw']:+.0f}, full circle at {best['full_circle']['dyaw']:+.0f} (z {best['full_circle']['z']:.1f}); keep the camera yaw")
    pb = pitch[int(np.argmin([r["loss"] for r in pitch]))]
    out = dict(camera_yaw_deg=cam_yaw, head="3DPS render_factorized (T23)", points=N_POINTS, fine=fine, coarse=coarse, pitch=pitch, refit_loss=refit,
               best=best, best_pitch_deg=pb["pitch"], refit_best_dyaw=float(rb), verdict=verdict, recommended=dict(dyaw_deg=best["loss"]["dyaw"] if measured else 0.0, pitch_deg=pb["pitch"] if measured else 0.0),
               seconds=time.time() - t0)
    os.makedirs(os.path.join(DATA, "calib_S1"), exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "calib_S1", "mount_yaw_v2.json"), "w"), indent=1)
    print("verdict:", verdict, f"| best pitch {pb['pitch']:+.0f} | refit best dyaw {rb} | {time.time()-t0:.0f} s")
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    a = ax[0, 0]; a.plot(FINE, L, "C0", label="static loss (lower = better)"); a.axvline(best["loss"]["dyaw"], color="C0", ls="--", lw=1)
    a.axvline(0, color="k", lw=1); a.set_xlabel("yaw offset from the camera yaw [deg]"); a.set_ylabel("loss"); a2 = a.twinx()
    a2.plot(FINE, CP, "C1", label="pair-mean profile corr"); a2.plot(FINE, CI, "C2", label="pair-image corr"); a2.set_ylabel("correlation")
    a.set_title(f"fine sweep: loss best {best['loss']['dyaw']:+.0f} deg (z {best['loss']['z']:.1f}); profile corr best {best['corr_profile']['dyaw']:+.0f}; image corr best {best['corr_image']['dyaw']:+.0f}", fontsize=10)
    a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8); a.grid(alpha=.3)
    a = ax[0, 1]; a.plot(COARSE, Lc, "C0.-"); a.axvspan(-60, 60, color="0.9"); a.axvline(0, color="k", lw=1)
    a.set_xlabel("yaw offset [deg], full circle (10 deg steps)"); a.set_ylabel("loss"); a.set_title(f"full circle: best {best['full_circle']['dyaw']:+.0f} deg (z {best['full_circle']['z']:.1f}); grey = fine-sweep range", fontsize=10); a.grid(alpha=.3)
    a = ax[1, 0]; a.plot([r["pitch"] for r in pitch], [r["loss"] for r in pitch], "C0o-"); a.set_xlabel(f"pitch [deg] at yaw offset {best['loss']['dyaw']:+.0f}"); a.set_ylabel("loss")
    a2 = a.twinx(); a2.plot([r["pitch"] for r in pitch], [r["corr_profile"] for r in pitch], "C1s--", label="profile corr"); a2.legend(fontsize=8)
    a.set_title(f"pitch sweep: best {pb['pitch']:+.0f} deg", fontsize=10); a.grid(alpha=.3)
    a = ax[1, 1]; ks = sorted(refit, key=float); a.bar([f"{float(k):+.0f}" for k in ks], [refit[k] for k in ks], color="C3", alpha=.8)
    a.plot([f"{float(k):+.0f}" for k in ks], [L[int(np.argmin(np.abs(FINE - float(k))))] for k in ks], "ko", label="before refit (uniform concrete)")
    a.set_xlabel("yaw offset [deg] (5 best + camera yaw)"); a.set_ylabel("loss"); a.set_title(f"{REFIT_STEPS}-step per-point material refit at fixed pose: best at {float(rb):+.0f} deg", fontsize=10); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(f"node_1 Vayyar mount yaw/pitch sweep v2 through the 3DPS-routed SFCW head (uniform concrete, {N_POINTS} pts)\n{verdict}", fontsize=11)
    fig.tight_layout()
    viz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "render_vs_gt"); os.makedirs(viz, exist_ok=True)
    fig.savefig(os.path.join(viz, "mount_yaw_sweep_v2.png"), dpi=140)
    with open(os.path.join(viz, "captions.txt"), "a") as fh:
        fh.write(f"mount_yaw_sweep_v2.png: node_1 Vayyar mount yaw/pitch sweep through the T23 3DPS-routed SFCW head (uniform concrete). {verdict}. Best pitch {pb['pitch']:+.0f} deg; {REFIT_STEPS}-step material refit best at {float(rb):+.0f} deg. Replaces mount_yaw_sweep.png (deprecated angle-blind model).\n")
    print("saved figure + json")


if __name__ == "__main__":
    main()
