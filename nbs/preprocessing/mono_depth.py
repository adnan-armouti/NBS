"""Monocular depth backends.

  'midas'        : MiDaS DPT (RELATIVE inverse depth; ported from 3DGPS/phase56).
  'dav2_metric'  : Depth-Anything-V2-Metric-Indoor (METRIC meters, via transformers).

Metric depth removes the affine ambiguity, so the per-clip depth scale recovered by
align.py becomes a stable rig/range-resolution constant rather than per-clip noise.
Run in an env with torch+timm (+transformers for dav2), e.g. ~/.conda/envs/mmir.
"""
import numpy as np


_MIDAS = {"model": None, "transform": None, "device": None}
_DAV2 = {"pipe": None, "name": None}


def _load(model_name="DPT_Large"):
    import torch
    if _MIDAS["model"] is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True).to(dev).eval()
        tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = tf.dpt_transform if "DPT" in model_name else tf.small_transform
        _MIDAS.update(model=m, transform=transform, device=dev)
    return _MIDAS


def disparity(rgb, model_name="DPT_Large"):
    """rgb: (H,W,3) uint8 RGB. Returns disparity (H,W) float32 (larger = nearer)."""
    import torch
    M = _load(model_name)
    with torch.no_grad():
        inp = M["transform"](rgb).to(M["device"])
        pred = M["model"](inp)
        disp = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
    return disp.cpu().numpy().astype(np.float32)


def relative_depth(rgb, model_name="DPT_Large", eps=1e-3):
    """Convert MiDaS disparity to a positive RELATIVE depth map (up to scale).

    depth_rel = 1 / (disp_norm + eps), with disp min-shifted to >=0.
    """
    disp = disparity(rgb, model_name)
    d = disp - disp.min()
    d = d / (d.max() + 1e-9)               # disp in [0,1], larger=nearer
    return (1.0 / (d + eps)).astype(np.float32)


_DAV2_MODELS = {
    "Small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def metric_depth_dav2(rgb, size="Small"):
    """METRIC depth (meters) via Depth-Anything-V2-Metric-Indoor (transformers)."""
    import torch
    from transformers import pipeline
    name = _DAV2_MODELS[size]
    if _DAV2["pipe"] is None or _DAV2["name"] != name:
        dev = 0 if torch.cuda.is_available() else -1
        _DAV2.update(pipe=pipeline("depth-estimation", model=name, device=dev), name=name)
    from PIL import Image
    out = _DAV2["pipe"](Image.fromarray(rgb))
    pd = out["predicted_depth"]
    d = pd.detach().cpu().numpy() if hasattr(pd, "detach") else np.asarray(pd)
    d = np.squeeze(d).astype(np.float32)
    if d.shape != rgb.shape[:2]:                # ensure full-res
        import cv2
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return d


def get_depth(rgb, backend="dav2_metric", **kw):
    """Dispatch: returns (depth, is_metric)."""
    if backend == "midas":
        return relative_depth(rgb, model_name=kw.get("model", "DPT_Large")), False
    if backend == "dav2_metric":
        return metric_depth_dav2(rgb, size=kw.get("size", "Small")), True
    raise ValueError(f"unknown depth backend {backend}")


def read_rgb(png_path):
    """Read a PNG as (H,W,3) uint8 RGB. Prefer cv2 (BGR->RGB), fall back to imageio."""
    try:
        import cv2
        bgr = cv2.imread(png_path)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    import imageio
    img = imageio.imread(png_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return img[..., :3]
