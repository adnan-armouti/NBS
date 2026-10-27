"""Index ALL clips of a scene into a manifest, tying each to the shared static
scaffold + its radar/pose + a frozen subject-disjoint split.

Given the moving human, the static geometry is a per-SCENE constant (build_scaffold),
so every clip points at the same scaffold; the human is per-clip dynamic content
(its motion is in the 17 radar frames + the per-clip 2D pose). This avoids 33k
redundant, human-corrupted per-clip clouds.

  python -m preprocessing.build_manifest --scene Scene1
Writes data/XRF55/manifest_<scene>.csv
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    ap.add_argument("--test-frac-subjects", type=float, default=0.2)
    a = ap.parse_args()

    mmw = IO.list_mmwave(a.scene)
    scaffold = os.path.join(C.NBS_DATA_OUT, a.scene, "scene", "pcl.npy")
    has_scaffold = os.path.exists(scaffold)

    rows = []
    subjects = sorted({IO.parse_key(p)["subject"] for p in mmw})
    n_test = max(1, int(round(len(subjects) * a.test_frac_subjects)))
    test_subjects = set(subjects[-n_test:])          # last subjects held out (disjoint)

    n_vid = n_pose = 0
    for p in mmw:
        k = IO.parse_key(p)
        vid = IO.video_path_for(p, a.scene)
        pose = os.path.join(C.XRF55_ROOT, "pose", "XRF55Videos_Res", "open",
                            a.scene.lower(), k["subject"], f"{k['subject']}_{k['action']}.json")
        has_v, has_p = os.path.exists(vid), os.path.exists(pose)
        n_vid += has_v; n_pose += has_p
        rows.append({
            "clip": f"{k['subject']}_{k['action']}_{k['trial']}",
            "subject": k["subject"], "action": k["action"], "trial": k["trial"],
            "mmwave": p, "video": vid if has_v else "", "pose": pose if has_p else "",
            "scaffold": scaffold if has_scaffold else "",
            "split": "test" if k["subject"] in test_subjects else "train",
        })

    out = os.path.join(C.NBS_DATA_OUT, f"manifest_{a.scene}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_train = sum(r["split"] == "train" for r in rows)
    print(f"[manifest] {len(rows)} clips | {len(subjects)} subjects "
          f"(test={sorted(test_subjects)}) | train {n_train} / test {len(rows)-n_train}")
    print(f"  videos present: {n_vid}/{len(rows)} | poses present: {n_pose}/{len(rows)} "
          f"| scaffold: {'yes' if has_scaffold else 'MISSING'}")
    print(f"  actions: {len(set(r['action'] for r in rows))} | -> {out}")


if __name__ == "__main__":
    main()
