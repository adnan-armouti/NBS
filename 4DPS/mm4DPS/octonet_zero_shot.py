"""T20 pilot: zero-shot cross-band evaluation of a per-point material trained on ONE band.

Protocol (the project's goal, user 2026-10-28): train on band X alone (the v4 single-band fits of `octonet_eval_panels`,
per-point six-parameter material), FREEZE the material, render band Y. Only Y's receiver nuisances are fitted on Y
(absolute gain, noise-floor scale and range correction: receiver properties, not scene properties; the coupling block
is measured, not fitted), with the material untouched. Three arms per (X -> Y):
    transfer   : X-trained material, Y nuisances fitted
    baseline   : uniform ITU concrete (the init), Y nuisances fitted
    upper bound: Y-trained material (the v4 fit of Y), Y nuisances fitted
What transfers per point: sigma_h, l_c, tau, thickness as they are (frequency-independent); eps from the per-point ITU
(a, c) with the init class exponents (concrete b = 0, d = 0.78), evaluated at Y's centre frequency.

    cd 4DPS && OCTONET_PANELS_TAG=v4 python -m mm4DPS.octonet_zero_shot --pairs sfcw:uwb,uwb:sfcw --seeds 0,1,2
Outputs $NBS_DATA_OUT/zero_shot_S1_<tag>/results.json (+ per-arm metrics).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from mm4DPS import octonet_fit_multiband as MB
from mm4DPS import octonet_eval_panels as EP
from mm4DPS import octonet_noise as ON
from method.freq_material import ITU_GT

TAG = os.environ.get("OCTONET_PANELS_TAG", "v4")
RUNS = os.path.join(MB.DATA_OUT, f"eval_panels_S1_{TAG}")
OUT = os.path.join(MB.DATA_OUT, f"zero_shot_S1_{TAG}")


def material_from_run(band: str, seed: int, dev, tag: str = TAG) -> dict:
    """The saved material of the panel fit <band>_tight_s<seed> of panel set ``tag`` as a heads material dict (abcd per point +
    roughness block; default roughness when the run saved none, e.g. a grid fit)."""
    z = np.load(os.path.join(MB.DATA_OUT, f"eval_panels_S1_{tag}", f"{band}_tight_s{seed}.npz"))
    mat = dict(abcd=torch.as_tensor(z["abcd"], dtype=torch.float64, device=dev), rho=None)
    if "physics" in z.files:
        ph = torch.as_tensor(z["physics"], dtype=torch.float64, device=dev); mat.update(sigma_h=ph[:, 2], l_c=ph[:, 3], tau=ph[:, 4], thickness=ph[:, 5])
    return mat, z["xyz"]


def fit_nuisance_only(band: str, cloud, material, sensors, targets, dev, steps: int, lr: float):
    nu = MB.BandNuisance(band, dev, True, True); nu.init(cloud, material, sensors, targets)
    opt = torch.optim.Adam(nu.parameters(), lr=lr); best = (float("inf"), None)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = MB.band_loss(band, cloud, material, sensors, targets, nuisance=nu)
        loss.backward(); opt.step()
        if float(loss) < best[0]: best = (float(loss), (nu.log_gain.detach().clone(), nu.floor.state_dict()))
    with torch.no_grad(): nu.log_gain.copy_(best[1][0]); nu.floor.load_state_dict(best[1][1])
    return nu, best[0]


def evaluate(band, cloud, material, sensors, targets, nu, tnp):
    rend = EP.render_all(band, cloud, material, sensors, targets, nu)
    return EP.metrics(band, rend, tnp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="sfcw:uwb,uwb:sfcw"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=150, help="nuisance-only Adam steps on the evaluation band"); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--points", type=int, default=20000); ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-tag", default=TAG, help="panel set the TRAINED (transferred) materials come from"); ap.add_argument("--eval-tag", default=TAG, help="panel set of the evaluation band's own fit (upper bound)")
    ap.add_argument("--out-tag", default=None)
    a = ap.parse_args(argv); dev = torch.device(a.device); out_dir = OUT if a.out_tag is None else os.path.join(MB.DATA_OUT, f"zero_shot_S1_{a.out_tag}"); os.makedirs(out_dir, exist_ok=True)
    results = []
    for pair in a.pairs.split(","):
        train, ev = pair.split(":")
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            cloud, material0, sensors = MB.setup(dev, a.points, seed); sensors = MB.sensors_without_gains(sensors)
            targets = MB.load_targets(dev, sfcw_guard=ON.guard_bins("sfcw", True), uwb_guard=ON.guard_bins("uwb", True)); tnp = EP.targets_np(targets)
            mat_x, xyz_x = material_from_run(train, seed, dev, a.train_tag); mat_y, xyz_y = material_from_run(ev, seed, dev, a.eval_tag)
            for nm, xyz in (("train", xyz_x), ("eval", xyz_y)):
                if not np.allclose(xyz, cloud["xyz"].cpu().numpy(), atol=1e-6): raise RuntimeError(f"{nm} material's cloud differs from setup(seed={seed}) -- same points/seed required")
            mat_c = dict(abcd=torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=dev).expand(cloud["xyz"].shape[0], 4).contiguous(), rho=None)
            # the SFCW channel gains are receiver properties of the Vayyar: fitted on Y when Y is SFCW (part of its nuisances), via a short joint fit
            row = dict(train=train, eval=ev, seed=seed, train_tag=a.train_tag, eval_tag=a.eval_tag)
            for arm, mat in (("transfer", mat_x), ("baseline", mat_c), ("upper", mat_y)):
                sens = sensors
                if ev == "sfcw":
                    g_tx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=True); g_rx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=True)
                    nu = MB.BandNuisance(ev, dev, True, True); nu.init(cloud, mat, sensors, targets)
                    opt = torch.optim.Adam([{"params": nu.parameters(), "lr": a.lr}, {"params": [g_tx, g_rx], "lr": a.lr}]); best = (float("inf"), None)
                    for step in range(a.steps):
                        opt.zero_grad(set_to_none=True); sens = MB.sensors_with_gains(sensors, g_tx, g_rx)
                        loss, _ = MB.band_loss(ev, cloud, mat, sens, targets, nuisance=nu); loss = loss + 0.01 * ((g_tx ** 2).mean() + (g_rx ** 2).mean())
                        loss.backward(); opt.step()
                        if float(loss) < best[0]: best = (float(loss), (nu.log_gain.detach().clone(), nu.floor.state_dict(), g_tx.detach().clone(), g_rx.detach().clone()))
                    with torch.no_grad(): nu.log_gain.copy_(best[1][0]); nu.floor.load_state_dict(best[1][1])
                    sens = MB.sensors_with_gains(sensors, best[1][2], best[1][3]); L = best[0]
                else:
                    nu, L = fit_nuisance_only(ev, cloud, mat, sensors, targets, dev, a.steps, a.lr)
                m = evaluate(ev, cloud, mat, sens, targets, nu, tnp)
                row[arm] = dict(loss=L, metrics=m, nuisance=nu.state())
                key = "corr_image_far" if ev == "sfcw" else "corr_ge1m"
                print(f"  {train}->{ev} seed {seed} {arm:8s}: loss {L:.4f}  {key} {m[key]:.3f}" + (f"  19cm {m['corr_image_far_5bin']:.3f} profile {m['corr_pair_mean_profile']:.3f}" if ev == "sfcw" else f"  39cm {m['corr_ge1m_61bin']:.3f} 1-9m {m['corr_1_9m']:.3f}"), flush=True)
            row["seconds"] = time.time() - t0; results.append(row)
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    # summary
    print("\nsummary (mean over seeds):")
    for pair in a.pairs.split(","):
        train, ev = pair.split(":"); rows = [r for r in results if r["train"] == train and r["eval"] == ev]
        key = "corr_image_far" if ev == "sfcw" else "corr_ge1m"
        for arm in ("baseline", "transfer", "upper"):
            v = np.array([r[arm]["metrics"][key] for r in rows]); L = np.array([r[arm]["loss"] for r in rows])
            print(f"  {train}->{ev} {arm:8s}: {key} {v.mean():.3f} +- {v.std():.3f}   loss {L.mean():.4f} +- {L.std():.4f}")


if __name__ == "__main__":
    main()
