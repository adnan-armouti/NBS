"""Iteration-0 check of the UWB head through the 3DPS rasterizer (UWB mode) on the calibrated Scene-1 scaffold vs the
pooled UWB target: log-envelope correlation over 0.3-9.9, 0.3-4 and 4-9.9 m, and the band loss, compared with the
numbers of the old point-sum head (viz/render_vs_gt_v2/README.md: iter-0 loss 1.18, corr -0.09).
Run: cd nbs && CUDA_VISIBLE_DEVICES=6 python -m method.uwb_3dps_render_check
"""
import json, os, sys, math
import numpy as np, torch
from preprocessing import octonet_config as C
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda:0"
z = np.load(os.path.join(DATA, "uwb_target_S1", "target_pool5.npz")); env_t = z["env"]; rel = np.clip(z["rel"], 0, 1); mask = z["mask"].astype(bool); r = z["range_m"]
cloud, material, _ = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=30000)
ext = json.load(open(os.path.join(DATA, "calib_S1", "extrinsics.json"))); sensor = OH.sensor_from_extrinsics(ext, "uwb")
with torch.no_grad():
    rf, env = OH.uwb_head(cloud, material, sensor)
env = env.cpu().numpy()
def logn(x, m): a = np.log(x[m] + 1e-30); return a - np.median(a)
g = logn(env_t, mask); e = logn(env, mask); w = rel[mask]
loss = float(np.sum(w * np.abs(e - g)) / np.sum(w))
def corr(sel):
    m = mask & sel; return float(np.corrcoef(logn(env_t, m), logn(env, m))[0, 1])
res = dict(loss_iter0=loss, corr_all=corr(np.ones_like(mask)), corr_0p3_4m=corr(r < 4.0), corr_4_9p9m=corr(r >= 4.0),
           rmse_db_all=float(np.sqrt(np.mean((20 / math.log(10) * (e - g)) ** 2))), peak_env_range_m=float(r[int(np.argmax(env))]))
print("UWB 3DPS head, iteration 0 (uniform concrete):", json.dumps(res, indent=1))
print("old point-sum head (README): loss 1.184, corr_all -0.089, corr 0.3-4 m 0.20, corr 4-9.9 m -0.23")
json.dump(res, open(os.path.join(DATA, "uwb_3dps_render_check.json"), "w"), indent=1)
