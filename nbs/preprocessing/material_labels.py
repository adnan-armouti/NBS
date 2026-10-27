"""Real-data material GROUND TRUTH for evaluation (eval-only; never fed to training).

Run off-the-shelf semantic segmentation on an XRF55 RGB frame, map indoor classes to ITU
materials, and project the per-pixel labels onto 3D radar-frame points via the Stage-1
camera->radar rig + metric depth. The result is a labelled point cloud used to score
RF-only material recovery (a cross-modal check; RGB is independent of RF).

  python -m preprocessing.material_labels --scene Scene1 --index 0
Run in the mmir env (torch + transformers).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G

# ADE20K class name -> ITU material (indoor subset; extend as needed)
ADE_TO_MATERIAL = {
    "wall": "concrete", "building": "concrete", "column": "concrete", "pillar": "concrete",
    "floor": "wood", "flooring": "wood", "rug": "wood", "carpet": "wood",
    "ceiling": "plasterboard",
    "windowpane": "glass", "window": "glass", "glass": "glass", "screen": "glass",
    "door": "wood", "cabinet": "wood", "table": "wood", "desk": "wood", "chair": "wood",
    "shelf": "wood", "bookcase": "wood", "wardrobe": "wood", "chest of drawers": "wood",
    "person": "__human__", "curtain": "__skip__", "blind": "__skip__",
}
MATERIALS = ["concrete", "wood", "plasterboard", "glass"]   # eval label set


def segment(rgb):
    """Per-pixel ADE20K material id (-1 = unknown/skip/human). Uses transformers Segformer."""
    import torch
    from transformers import pipeline
    seg = pipeline("image-segmentation", model="nvidia/segformer-b4-finetuned-ade-512-512",
                   device=0 if torch.cuda.is_available() else -1)
    from PIL import Image
    out = seg(Image.fromarray(rgb))
    H, W = rgb.shape[:2]
    mat = np.full((H, W), -1, dtype=np.int64)
    for seg_i in out:                                  # each: {label, mask(PIL)}
        name = seg_i["label"].lower()
        material = ADE_TO_MATERIAL.get(name)
        if material in MATERIALS:
            m = np.array(seg_i["mask"]) > 0
            mat[m] = MATERIALS.index(material)
        elif material == "__human__":
            mat[np.array(seg_i["mask"]) > 0] = -2      # human -> drop
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    from preprocessing import mono_depth as MD
    from preprocessing.prep_scene import clip_depth_mid, clips_with_media

    mmw = clips_with_media(a.scene)[a.index]
    clip = "_".join(IO.parse_key(mmw).values())
    rig = json.load(open(os.path.join(C.NBS_DATA_OUT, f"{a.scene}_rig.json")))
    ext = C.Extrinsics(); ext.yaw_deg, ext.cam_x, ext.cam_y, ext.cam_z, ext.pitch_deg = (
        rig["yaw_deg"], rig["cam_x"], rig["cam_y"], rig["cam_z"], rig["pitch_deg"])
    K = C.camera_K()

    depth, _ = clip_depth_mid(mmw, a.scene)            # metric depth (pose-matched middle frame)
    idx = len(IO.load_pose(mmw, a.scene)) // 2
    png = os.path.join(C.NBS_DATA_OUT, clip, "views", f"frame_{idx:05d}.png")
    rgb = MD.read_rgb(png)
    matpix = segment(rgb)                              # (H,W) material id

    # unproject (a=1 metric) + carry the per-pixel material label, drop human/unknown
    H, W = depth.shape
    vs, us = np.mgrid[0:H:a.stride, 0:W:a.stride]
    pts0, rays, _ = G.unproject_bases(depth, K, stride=a.stride)
    lab = matpix[vs, us].ravel()
    d = depth[vs, us].ravel()
    good = (np.isfinite(d) & (d > 0))
    lab = lab[good]
    import torch
    pr = G.transform_pose(pts0, ext).cpu().numpy() if torch.is_tensor(pts0) else G.transform_pose(pts0, ext)
    pr = pr if isinstance(pr, np.ndarray) else pr.cpu().numpy()
    keep = lab >= 0
    pr, lab = pr[keep], lab[keep]
    r = np.linalg.norm(pr, axis=1); rk = (r >= 0.5) & (r < 256 * 0.076)
    pr, lab = pr[rk], lab[rk]

    out = os.path.join(C.NBS_DATA_OUT, clip)
    np.save(os.path.join(out, "material_gt.npy"), np.concatenate([pr, lab[:, None]], 1).astype(np.float32))
    counts = {MATERIALS[i]: int((lab == i).sum()) for i in range(len(MATERIALS))}
    json.dump({"clip": clip, "materials": MATERIALS, "counts": counts, "n_points": int(len(lab)),
               "note": "EVAL-ONLY material GT (RGB semantic seg -> ITU material -> radar frame)"},
              open(os.path.join(out, "material_gt.json"), "w"), indent=2)
    print(f"[material GT] {clip}: {len(lab)} labelled points  counts={counts}")

    # viz: RGB material overlay + top-down labelled points
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ov = rgb.copy().astype(float)
    cols = np.array([[200, 80, 80], [80, 160, 80], [80, 80, 200], [200, 200, 80]])
    for i in range(len(MATERIALS)):
        ov[matpix == i] = 0.5 * ov[matpix == i] + 0.5 * cols[i]
    ax[0].imshow(ov.astype(np.uint8)); ax[0].set_title(f"{clip} RGB + material seg\n{MATERIALS}"); ax[0].axis("off")
    sc = ax[1].scatter(pr[:, 0], pr[:, 1], c=lab, cmap="tab10", s=2, vmin=0, vmax=9)
    ax[1].scatter([0], [0], c="k", marker="^", s=80); ax[1].set_title("labelled points (top-down, radar frame)")
    ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y range (m)"); ax[1].axis("equal")
    fig.tight_layout(); fig.savefig(os.path.join(out, "material_gt.png"), dpi=90); plt.close(fig)
    print(f"saved {out}/material_gt.npy + .png")


if __name__ == "__main__":
    main()
