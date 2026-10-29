"""Joint-training panels (viz/render_vs_gt_joint_v1): ALL bands optimise ONE per-point scene material.

  A_all3          SFCW + UWB + WiFi train together (joint loss = sum_b L_b / L_b(start)); panels + tables per band,
                  with the single-band v4 fit of each band as the reference.
  B_holdout_<y>   the other two bands train together; band y is HELD OUT and evaluated zero-shot with the T20 protocol
                  (`octonet_zero_shot.fit_eval_nuisances`: only y's receiver nuisances are fitted on y, the material is
                  frozen): arms transfer (the two-band material) / baseline (uniform ITU concrete) / upper (y's own v4 fit).
Model = the v4 model: per-point six-parameter 3DPS material (tight ITU bounds), learned noise floor + coupling block for
SFCW and UWB, 20 + 20 SFCW channel gains; WiFi has no nuisance model and a median-normalised loss (its head reads the
permittivity only). Joint optimisation: Adam lr 0.02, 600 steps (WiFi alone needs 0.01 / 600; SFCW/UWB 0.03 / 400).

    cd 4DPS && CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_joint_panels fit --set A_all3 --seeds 0,1,2
    python -m mm4DPS.octonet_joint_panels figures            # every set found -> viz/render_vs_gt_joint_v1/<set>/ + README.md
Outputs $NBS_DATA_OUT/joint_panels_S1_v1/<set>/<band>_s<seed>.{json,npz}.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from mm4DPS import octonet_fit_multiband as MB
from mm4DPS import octonet_eval_panels as EP
from mm4DPS import octonet_noise as ON
from mm4DPS import octonet_zero_shot as ZS

TAG = os.environ.get("OCTONET_JOINT_TAG", "v1")
OUT_ROOT = os.path.join(MB.DATA_OUT, f"joint_panels_S1_{TAG}")
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", f"render_vs_gt_joint_{TAG}")
REF_TAG = "v4"
BANDS = ("sfcw", "uwb", "wifi")
SETS = {"A_all3": (("sfcw", "uwb", "wifi"), None),
        "B_holdout_sfcw": (("uwb", "wifi"), "sfcw"), "B_holdout_uwb": (("sfcw", "wifi"), "uwb"), "B_holdout_wifi": (("sfcw", "uwb"), "wifi")}
KEY = dict(sfcw=("corr_image_far", "pair image ≥ 0.9 m (ceiling 0.89)"), uwb=("corr_ge1m", "envelope ≥ 1 m (ceiling 0.95)"), wifi=("corr", "|CSI| all receivers"))


def metric(m: dict, band: str, key: str | None = None):
    k = key or KEY[band][0]
    return m["all"][k] if band == "wifi" else m[k]


def save_band(out_dir, band, seed, cloud, abcd, mat1, tnp, r0, r1, extra: dict):
    save = dict(xyz=cloud["xyz"].cpu().numpy(), abcd=abcd, physics=MB.physics_columns(mat1, MB.BAND_FC_GHZ[band]), mask=(tnp[band]["mask"] if band != "wifi" else np.zeros(0)))
    if band == "wifi":
        for k in (1, 2, 3, 4): save[f"r0_{k}"] = r0[k]; save[f"r1_{k}"] = r1[k]
    else:
        save["r0"] = r0; save["r1"] = r1
    gm = extra.get("gain_metrics") or {}
    if gm.get("enabled"):
        save.update(gain_tx_db=np.asarray(gm["tx_db"]), gain_rx_db=np.asarray(gm["rx_db"]), gt_gain_tx_db=np.asarray(gm["gt_tx_db"]), gt_gain_rx_db=np.asarray(gm["gt_rx_db"]))
    np.savez(os.path.join(out_dir, f"{band}_s{seed}.npz"), **save)
    json.dump(extra, open(os.path.join(out_dir, f"{band}_s{seed}.json"), "w"), indent=1)


def cmd_fit(a):
    train, held = SETS[a.set]; out_dir = os.path.join(OUT_ROOT, a.set); os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(a.device); B = EP.set_bounds("tight")
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time(); tag = f"{a.set}_s{seed}"
        print(f"=== {tag}: train {train}, held {held}, steps {a.steps}, lr {a.lr}", flush=True)
        res, grid, abcd, cloud, sensors, targets = MB.fit(train, dev, a.points, a.steps, a.lr, 64, 1e-3, seed, tag=tag, gains=True, gain_l2=0.01, patience=0,
                                                          material="per-point", l2=a.l2, noise_model=True, coupling=True)
        nuis = res.pop("nuis"); res.pop("log", None)
        tnp = EP.targets_np(targets); sensors0 = MB.sensors_without_gains(sensors)
        mat0 = ZS.concrete_material(cloud["xyz"].shape[0], dev)
        mat1 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in grid.material_at().items()}      # frozen: no graph into the held-out nuisance fit
        common = dict(set=a.set, train_bands=list(train), held_band=held, seed=seed, material="per-point", bounds="tight", bounds_values={k: (list(v) if isinstance(v, tuple) else v) for k, v in B.items()},
                      steps=res["steps"], best_step=res["best_step"], runtime_s=res["runtime_s"], lr=a.lr, l2=a.l2, guards=res["guards"], dispersion=MB.dispersion_summary(abcd), joint_start=res["start"], joint_end=res["end"])
        # training bands: GT / iter 0 / trained, as in the single-band panels
        for b in train:
            nu0 = None
            if b in nuis:
                nu0 = MB.BandNuisance(b, dev, True, True); nu0.init(cloud, mat0, sensors0, targets)
            r0 = EP.render_all(b, cloud, mat0, sensors0, targets, nu0); r1 = EP.render_all(b, cloud, mat1, sensors, targets, nuis.get(b))
            m0, m1 = EP.metrics(b, r0, tnp), EP.metrics(b, r1, tnp)
            gm = EP.gain_metrics(res.get("gains"), tnp["sfcw"]) if b == "sfcw" else dict(enabled=False)
            extra = dict(common, band=b, role="train", loss_iter0=res["start"][b], loss_trained=res["end"][b], metrics_iter0=m0, metrics_trained=m1, gain_metrics=gm,
                         nuisance=res["nuisance"].get(b), gains=res.get("gains"))
            save_band(out_dir, b, seed, cloud, abcd, mat1, tnp, r0, r1, extra)
            print(f"  {tag} {b}: loss {res['start'][b]:.4f} -> {res['end'][b]:.4f}  {KEY[b][0]} {metric(m0, b):.3f} -> {metric(m1, b):.3f}", flush=True)
        # held-out band: T20 arms, nuisances only
        if held:
            mat_up, xyz_up = ZS.material_from_run(held, seed, dev, REF_TAG)
            assert np.allclose(xyz_up, cloud["xyz"].cpu().numpy(), atol=1e-6), "v4 reference material was fitted on a different cloud"
            arms = {}; rends = {}
            for arm, mat in (("baseline", mat0), ("transfer", mat1), ("upper", mat_up)):
                fe = ZS.fit_eval_nuisances(held, cloud, mat, sensors0, targets, dev, a.eval_steps, 0.03)
                m, rend = ZS.evaluate(held, cloud, mat, fe["sensors"], targets, fe["nu"], tnp)
                arms[arm] = dict(loss=fe["loss"], metrics=m, gain_metrics=(EP.gain_metrics(fe["gains"], tnp["sfcw"]) if held == "sfcw" else dict(enabled=False)),
                                 nuisance=(fe["nu"].state() if fe["nu"] is not None else None))
                rends[arm] = rend
                print(f"  {tag} held {held} {arm:8s}: loss {fe['loss']:.4f}  {KEY[held][0]} {metric(m, held):.3f}", flush=True)
            extra = dict(common, band=held, role="held", arms=arms, loss_iter0=arms["baseline"]["loss"], loss_trained=arms["transfer"]["loss"], loss_upper=arms["upper"]["loss"],
                         metrics_iter0=arms["baseline"]["metrics"], metrics_trained=arms["transfer"]["metrics"], metrics_upper=arms["upper"]["metrics"], gain_metrics=arms["transfer"]["gain_metrics"])
            save_band(out_dir, held, seed, cloud, abcd, mat1, tnp, rends["baseline"], rends["transfer"], extra)
        print(f"  {tag}: {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- figures + tables
def load_set(name):
    d = os.path.join(OUT_ROOT, name); runs = {}
    if not os.path.isdir(d): return runs
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"): runs[f[:-5]] = json.load(open(os.path.join(d, f)))
    return runs


def ref_v4(band):
    d = os.path.join(MB.DATA_OUT, f"eval_panels_S1_{REF_TAG}"); out = []
    for s in (0, 1, 2):
        p = os.path.join(d, f"{band}_tight_s{s}.json")
        if os.path.exists(p): out.append(json.load(open(p)))
    return out


def ms(vals, fmt="{:.3f}"):
    v = np.array(vals, float); return (fmt.format(v.mean()) + " ± " + fmt.format(v.std())) if len(v) > 1 else (fmt.format(v[0]) if len(v) else "—")


def band_table_rows(band, runs_b):
    """Per-band detail rows (same columns as the v4 README), for train or held role (held: iter0 = baseline arm, trained = transfer arm)."""
    rows = []
    for k, v in runs_b.items():
        a, b = v["metrics_iter0"], v["metrics_trained"]
        if band == "sfcw":
            g = v.get("gain_metrics", {}) or {}
            gcell = f"{g['tx_std_db']:.2f}/{g['rx_std_db']:.2f} ({g['gt_tx_std_db']:.2f}/{g['gt_rx_std_db']:.2f}) | {g['corr_tx']:.2f} / {g['corr_rx']:.2f} / {g['corr_pairs']:.2f}" if g.get("enabled") else "— | —"
            rows.append(f"| {k} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['rmse_db']:.2f} → {b['rmse_db']:.2f} | {a['corr_image']:.3f} → {b['corr_image']:.3f} | {a['corr_image_far']:.3f} → **{b['corr_image_far']:.3f}** / {b['corr_image_far_5bin']:.3f} | {a['corr_pair_median']:.3f} → {b['corr_pair_median']:.3f} | {a['corr_pair_mean_profile']:.3f} → {b['corr_pair_mean_profile']:.3f} / {a['rmse_db_pair_mean_profile']:.2f} → {b['rmse_db_pair_mean_profile']:.2f} | {gcell} |")
        elif band == "uwb":
            rows.append(f"| {k} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['rmse_db']:.2f} → {b['rmse_db']:.2f} | {a['corr_all']:.3f} → {b['corr_all']:.3f} | {a['corr_ge1m']:.3f} → **{b['corr_ge1m']:.3f}** / {b['corr_ge1m_61bin']:.3f} | {b['corr_1_9m']:.3f} / {b['corr_1_9m_61bin']:.3f} | {a['corr_0p3_4m']:.3f} → {b['corr_0p3_4m']:.3f} | {a['corr_4_9p9m']:.3f} → {b['corr_4_9p9m']:.3f} |")
        else:
            cells = " | ".join(f"{a[f'rx{i}']['rmse_db']:.2f}→{b[f'rx{i}']['rmse_db']:.2f} / {a[f'rx{i}']['corr']:.2f}→{b[f'rx{i}']['corr']:.2f}" for i in (1, 2, 3, 4))
            rows.append(f"| {k} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['all']['rmse_db']:.2f} → {b['all']['rmse_db']:.2f} | {a['all']['corr']:.3f} → **{b['all']['corr']:.3f}** | {cells} |")
    hdr = {"sfcw": ["| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |", "|---|---|---|---|---|---|---|---|---|"],
           "uwb": ["| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m |", "|---|---|---|---|---|---|---|---|"],
           "wifi": ["| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |", "|---|---|---|---|---|---|---|---|"]}[band]
    return hdr + rows


def set_readme(name, runs, viz_dir):
    train, held = SETS[name]; lines = [f"# {name}: " + ("all three bands train one per-point material" if held is None else f"{' + '.join(b.upper() for b in train)} train one per-point material; {held.upper()} held out (zero-shot)"), ""]
    lines += ["Model: per-point six-parameter 3DPS material (tight ITU bounds, L2 0.05 to the init), learned noise floor + coupling block (SFCW, UWB), 20 + 20 SFCW channel gains; "
              "joint loss = Σ_b L_b / L_b(iteration 0); Adam lr 0.02, 600 steps; 3 seeds. 'iter 0' = uniform ITU concrete with the iteration-0 gain/floor. "
              "Reference = the single-band v4 fits (`viz/render_vs_gt_v4`). Metrics on the masked, median-normalised dB quantities (comparable with v4).", ""]
    captions = []
    # summary over training bands
    lines += ["## Training bands (mean ± std over seeds)", "", "| band | loss iter 0 | loss trained, joint | loss trained, single-band v4 | corr trained, joint | corr trained, single-band v4 | metric |", "|---|---|---|---|---|---|---|"]
    for b in train:
        rb = {k: v for k, v in runs.items() if v["band"] == b and v["role"] == "train"}
        if not rb: continue
        ref = ref_v4(b)
        lines.append(f"| {b.upper()} | {ms([v['loss_iter0'] for v in rb.values()])} | **{ms([v['loss_trained'] for v in rb.values()])}** | {ms([r['loss_trained'] for r in ref])} | **{ms([metric(v['metrics_trained'], b) for v in rb.values()])}** | {ms([metric(r['metrics_trained'], b) for r in ref])} | {KEY[b][1]} |")
    lines.append("")
    if held:
        rh = {k: v for k, v in runs.items() if v["band"] == held and v["role"] == "held"}
        if rh:
            base = np.array([metric(v["arms"]["baseline"]["metrics"], held) for v in rh.values()]); tr = np.array([metric(v["arms"]["transfer"]["metrics"], held) for v in rh.values()]); up = np.array([metric(v["arms"]["upper"]["metrics"], held) for v in rh.values()])
            gain = tr - base; frac = gain.mean() / max(up.mean() - base.mean(), 1e-9)
            verdict = "PASS" if (gain.mean() > 3 * (gain.std() + 1e-9) and frac >= 0.25) else "FAIL"
            lines += [f"## Held-out band {held.upper()}: zero-shot (T20 protocol: only {held.upper()}'s receiver nuisances fitted, material frozen)", "",
                      "| arm | material | loss | corr (" + KEY[held][1] + ") |", "|---|---|---|---|"]
            for arm, desc in (("baseline", "uniform ITU concrete"), ("transfer", f"trained on {' + '.join(x.upper() for x in train)}"), ("upper", f"{held.upper()}'s own single-band fit (v4)")):
                lines.append(f"| {arm} | {desc} | {ms([v['arms'][arm]['loss'] for v in rh.values()], '{:.4f}')} | **{ms([metric(v['arms'][arm]['metrics'], held) for v in rh.values()])}** |")
            lines += ["", f"Transfer − baseline = {gain.mean():+.3f} ± {gain.std():.3f} = {100 * frac:.0f} % of (upper − baseline); gate (> 3 σ and ≥ 25 %): **{verdict}**.", ""]
    # per-band detail tables + figures (best seed = lowest trained loss; held: lowest transfer loss)
    dev = torch.device("cpu")
    g = runs[next(iter(runs))]["guards"]; targets = MB.load_targets(dev, sfcw_guard=g["sfcw"], uwb_guard=g["uwb"]); tnp = EP.targets_np(targets)
    for b in list(train) + ([held] if held else []):
        rb = {k: v for k, v in runs.items() if v["band"] == b}
        if not rb: continue
        role = "held" if (held and b == held) else "train"
        lines += [f"## {b.upper()} ({'held out' if role == 'held' else 'training band'})", ""]
        if role == "held": lines += ["Rows: iter0 = the baseline arm (uniform concrete + fitted nuisances), trained = the transfer arm (two-band material + fitted nuisances).", ""]
        lines += band_table_rows(b, rb) + [""]
        bt = min(rb, key=lambda k: rb[k]["loss_trained"]); z = np.load(os.path.join(OUT_ROOT, name, bt + ".npz")); run = rb[bt]
        lab0 = "rendered, iter 0 (uniform concrete)" if role == "train" else "baseline: uniform concrete + fitted nuisances"
        lab1 = f"rendered, trained jointly ({' + '.join(x.upper() for x in train)})" if role == "train" else f"zero-shot: material from {' + '.join(x.upper() for x in train)}"
        fig_path = os.path.join(viz_dir, f"{b}_gt_iter0_trained.png" if role == "train" else f"{b}_heldout_baseline_transfer.png")
        {"sfcw": EP.fig_sfcw, "uwb": EP.fig_uwb, "wifi": EP.fig_wifi}[b](tnp, z, f"{name} {bt}", fig_path, label0=lab0, label1=lab1)
        captions.append(f"{os.path.basename(fig_path)}: {b.upper()} GT vs {lab0} vs {lab1} ({bt}); last column = residual.")
        EP.fig_material(b, z, run, f"{name} {bt}", os.path.join(viz_dir, f"{b}_material.png")); captions.append(f"{b}_material.png: the shared per-point material at {MB.BAND_FC_GHZ[b]} GHz ({bt}).")
        if b == "sfcw" and role == "train" and run.get("gain_metrics", {}).get("enabled"):
            EP.fig_gains(tnp, z, run, f"{name} {bt}", os.path.join(viz_dir, "sfcw_gains.png")); captions.append("sfcw_gains.png: fitted SFCW channel gains vs GT-derived.")
        d = run["dispersion"][f"{MB.BAND_FC_GHZ[b]}GHz"]
        lines += [f"Shared material at {MB.BAND_FC_GHZ[b]} GHz (best seed `{bt}`): ε′ median {d['eps_median']:.2f} [p5 {d['eps_p05']:.2f}, p95 {d['eps_p95']:.2f}], σ median {d['sigma_median']:.3f} S/m; nearest ITU class {run['dispersion']['nearest_itu_class']}.", ""]
    open(os.path.join(viz_dir, "README.md"), "w").write("\n".join(lines) + "\n"); open(os.path.join(viz_dir, "captions.txt"), "w").write("\n".join(captions) + "\n")
    return lines


def cmd_figures(a):
    os.makedirs(VIZ_ROOT, exist_ok=True)
    top = [f"# Joint training (render_vs_gt_joint_{TAG})", "", "A: all three bands optimise one per-point scene material. B: two bands train, the third is held out and evaluated zero-shot "
           "(only its receiver nuisances fitted; arms baseline / transfer / upper). Model and protocol: `4DPS/mm4DPS/octonet_joint_panels.py`, `octonet_zero_shot.py`. "
           "Reference single-band numbers: `viz/render_vs_gt_v4/`.", "", "| set | subfolder | training bands: loss trained (joint vs single v4) | held-out band: baseline → transfer → upper (corr) | gate |", "|---|---|---|---|---|"]
    for name in SETS:
        runs = load_set(name)
        if not runs: top.append(f"| {name} | — | no runs | | |"); continue
        viz_dir = os.path.join(VIZ_ROOT, name); os.makedirs(viz_dir, exist_ok=True)
        set_readme(name, runs, viz_dir)
        train, held = SETS[name]
        cells = []
        for b in train:
            rb = [v for v in runs.values() if v["band"] == b and v["role"] == "train"]; ref = ref_v4(b)
            cells.append(f"{b.upper()} {ms([v['loss_trained'] for v in rb])} (v4 {ms([r['loss_trained'] for r in ref])})")
        hcell = gate = ""
        if held:
            rh = [v for v in runs.values() if v["band"] == held and v["role"] == "held"]
            if rh:
                base = np.array([metric(v["arms"]["baseline"]["metrics"], held) for v in rh]); tr = np.array([metric(v["arms"]["transfer"]["metrics"], held) for v in rh]); up = np.array([metric(v["arms"]["upper"]["metrics"], held) for v in rh])
                gain = tr - base; frac = gain.mean() / max(up.mean() - base.mean(), 1e-9)
                hcell = f"{held.upper()}: {base.mean():.3f} → **{tr.mean():.3f}** → {up.mean():.3f}"; gate = ("PASS" if (gain.mean() > 3 * (gain.std() + 1e-9) and frac >= 0.25) else "FAIL") + f" ({100 * frac:.0f} % of the gap)"
        top.append(f"| {name} | [{name}/]({name}/README.md) | {'; '.join(cells)} | {hcell} | {gate} |")
    open(os.path.join(VIZ_ROOT, "README.md"), "w").write("\n".join(top) + "\n"); print("\n".join(top))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--set", required=True, choices=list(SETS)); f.add_argument("--seeds", default="0,1,2")
    f.add_argument("--steps", type=int, default=600); f.add_argument("--lr", type=float, default=0.02); f.add_argument("--l2", type=float, default=0.05)
    f.add_argument("--eval-steps", type=int, default=150, help="nuisance-only steps on the held-out band"); f.add_argument("--points", type=int, default=20000); f.add_argument("--device", default="cuda")
    sub.add_parser("figures")
    a = ap.parse_args(argv); {"fit": cmd_fit, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
