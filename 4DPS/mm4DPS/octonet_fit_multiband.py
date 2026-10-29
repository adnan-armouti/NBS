"""OctoNet single-band and joint material fits, 3DPS-style (per-point materials), with the ported 4DPS nuisance models.

Material (default ``per-point``, as 3DPS optimises it): ONE six-parameter material per scaffold point
[eps', eps'', sigma_h, l_c, tau, thickness]; eps', eps'' come from a per-point ITU-R P.2040 dispersion
eps'(f) = a f^b, sigma(f) = c f^d evaluated at the band centre, with (a, c) per point and the exponents (b, d) frozen at
the init class (a single band cannot identify them; ``free_exponents`` releases them for joint fits), and the four
roughness/slab parameters per point in the 3DPS raw space (``reparameterize_torch`` bounds). ``grid`` keeps the
Hofmann-style voxel field of T11 (4 ITU coefficients on a V^3 grid, trilinear) for the transfer experiments.
Nuisances (``mm4DPS.octonet_noise``, the 4DPS models): a learned receiver noise floor (measured envelope prior x learned
scale x DCT range correction, added in POWER under the magnitude) and the coherent near-field coupling block (added
after a learned absolute gain; the loss then sees the near bins the T4 guard removed). With them the SFCW/UWB losses are
log-magnitude L1 in absolute units; WiFi keeps the median-normalised magnitude-only loss (no envelope measured).
  SFCW : nbs/method/sfcw_target.sfcw_static_loss(kind='logmag', weight='xcorr'), 20 + 20 channel gains (D13)
  UWB  : log|hilbert(cir_rf)| vs log(target env), weighted by cross-recording reliability
  WiFi : log|CSI| per receiver/antenna/subcarrier vs log(mean |CSI|), HT40 notches masked, weighted by reliability
Joint loss = sum_b L_b / L_b(start).

  python -m mm4DPS.octonet_fit_multiband run --bands sfcw                        # per-point + noise model + coupling (defaults)
  python -m mm4DPS.octonet_fit_multiband run --bands uwb --material grid --noise-model off --coupling off   # the T11 grid, gated
  python -m mm4DPS.octonet_fit_multiband gate --held wifi --seeds 0,1,2
  python -m mm4DPS.octonet_fit_multiband report                                   # tables + viz/render_vs_gt/multiband_fit.png
Outputs under $NBS_DATA_OUT/fit_S1_multiband/.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch

from mm4DPS import octonet_heads as OH
from mm4DPS import octonet_noise as ON
from mm4DPS.octonet import render_inputs
from method import sfcw_target as ST
from method.freq_material import ITU_GT, eps_from_dispersion
from preprocessing import octonet_config as C, octonet_scene as S

DATA_OUT = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
OUT = os.path.join(DATA_OUT, "fit_S1_multiband")
# default scaffold: v1 (voxel PCA normals). v2 (planar normals, D16) is available via OCTONET_SCAFFOLD_DIR but trains WORSE
# without gains (0.735 vs 0.665) and equal within seed noise with gains (2026-10-29 A/B), so it is not the default.
SCAFFOLD_DIR = os.environ.get("OCTONET_SCAFFOLD_DIR", os.path.join(DATA_OUT, "4dps_scaffold_S1"))
EXTRINSICS = os.path.join(DATA_OUT, "calib_S1", "extrinsics.json")
BANDS = ("sfcw", "uwb", "wifi")
BAND_FC_GHZ = dict(wifi=5.18, uwb=7.0, sfcw=65.0)
EPS = 1e-12
GUARD_LO_SFCW = 24


# ----------------------------------------------------------------------------- material storage
def abcd_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """raw (...,4) -> bounded ITU (a,b,c,d) covering the P.2040 table."""
    a = 1.0 + 14.0 * torch.sigmoid(raw[..., 0])
    b = 0.2 * torch.tanh(raw[..., 1])
    c = torch.exp(math.log(1e-3) + torch.sigmoid(raw[..., 2]) * (math.log(2.0) - math.log(1e-3)))
    d = 1.5 * torch.sigmoid(raw[..., 3])
    return torch.stack([a, b, c, d], -1)


def raw_from_abcd(abcd) -> torch.Tensor:
    a, b, c, d = [float(v) for v in abcd]
    def logit(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
    return torch.tensor([logit((a - 1.0) / 14.0), math.atanh(max(min(b / 0.2, 0.999), -0.999)),
                         logit((math.log(max(c, 1.001e-3)) - math.log(1e-3)) / (math.log(2.0) - math.log(1e-3))), logit(d / 1.5)], dtype=torch.float64)


class VoxelGridMaterial(torch.nn.Module):
    """Hofmann-style voxel grid of raw material parameters, trilinear lookup at points (grid_sample)."""

    def __init__(self, xyz: torch.Tensor, voxels: int = 28, init_abcd=ITU_GT["concrete"], noise: float = 0.01, n_params: int = 4):
        super().__init__()
        lo = xyz.min(0).values - 0.05; hi = xyz.max(0).values + 0.05
        self.register_buffer("lo", lo); self.register_buffer("hi", hi)
        self.voxels = voxels
        base = raw_from_abcd(init_abcd).to(xyz.device)
        g = torch.zeros(1, n_params, voxels, voxels, voxels, dtype=torch.float64, device=xyz.device)
        g[0, :4] = base.view(4, 1, 1, 1)
        g = g + noise * torch.randn_like(g)
        self.grid = torch.nn.Parameter(g)

    def raw_at(self, xyz: torch.Tensor) -> torch.Tensor:
        u = (xyz - self.lo) / (self.hi - self.lo) * 2.0 - 1.0                          # (N,3) in [-1,1]
        coords = u[None, None, None, :, :].flip(-1)                                    # grid_sample wants (x,y,z) = (W,H,D) order
        out = torch.nn.functional.grid_sample(self.grid, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return out[0, :, 0, 0, :].T                                                    # (N,P)

    def material_at(self, xyz: torch.Tensor) -> dict:
        raw = self.raw_at(xyz)
        return dict(abcd=abcd_from_raw(raw[:, :4]), rho=None)

    def tv(self) -> torch.Tensor:
        g = self.grid
        return ((g[..., 1:, :, :] - g[..., :-1, :, :]) ** 2).mean() + ((g[..., :, 1:, :] - g[..., :, :-1, :]) ** 2).mean() + ((g[..., :, :, 1:] - g[..., :, :, :-1]) ** 2).mean()

    @property
    def voxel_m(self) -> float:
        return float(((self.hi - self.lo) / (self.voxels - 1)).max())


class PerPointMaterial(torch.nn.Module):
    """3DPS: one six-parameter material per point, frequency-aware. State = raw (a, c) per point [+ (b, d) when
    ``free_exponents``] through ``abcd_from_raw``'s bounds, and the raw 3DPS roughness block (sigma_h, l_c, tau,
    thickness) per point through ``reparameterize_torch``. ``material_at`` returns the heads' material dict."""

    def __init__(self, xyz: torch.Tensor, init_abcd=ITU_GT["concrete"], noise: float = 0.01, free_exponents: bool = False,
                 init_rough: dict | None = None, smooth_knn: float = 0.0, k: int = 8):
        super().__init__()
        from mm3DPS.rasterizer import reparameterize_torch
        n = xyz.shape[0]; dev = xyz.device
        self.smooth_knn = float(smooth_knn)
        if self.smooth_knn > 0:                       # graph Laplacian prior: neighbours on the same surface share a material
            self.register_buffer("nbr", knn_indices(xyz, k))
        base = raw_from_abcd(init_abcd).to(dev)
        self.raw_ac = torch.nn.Parameter(base[[0, 2]].expand(n, 2).clone() + noise * torch.randn(n, 2, dtype=torch.float64, device=dev))
        bd = base[[1, 3]].expand(n, 2).clone()
        if free_exponents: self.raw_bd = torch.nn.Parameter(bd)
        else: self.register_buffer("raw_bd", bd)
        r = dict(OH.SFCW_DEFAULT_ROUGHNESS); r.update(init_rough or {})
        raw_r = torch.tensor([math.log(r["sigma_h"]), math.log(r["l_c"]), math.log((r["tau"] - 0.05) / (0.95 - r["tau"])), math.log(r["thickness"])], dtype=torch.float64, device=dev)
        self.raw_rough = torch.nn.Parameter(raw_r.expand(n, 4).clone() + noise * torch.randn(n, 4, dtype=torch.float64, device=dev))
        self.register_buffer("raw_ac0", self.raw_ac.detach().clone()); self.register_buffer("raw_rough0", self.raw_rough.detach().clone())
        self._reparam = reparameterize_torch
        self.free_exponents = free_exponents

    def raw_abcd(self) -> torch.Tensor:
        return torch.stack([self.raw_ac[:, 0], self.raw_bd[:, 0], self.raw_ac[:, 1], self.raw_bd[:, 1]], -1)

    def material_at(self, xyz: torch.Tensor | None = None) -> dict:
        abcd = abcd_from_raw(self.raw_abcd())                                       # module-level: the panels' bounds switch applies
        six = self._reparam(torch.cat([torch.zeros_like(self.raw_rough[:, :2]), self.raw_rough], -1))
        return dict(abcd=abcd, sigma_h=six[:, 2], l_c=six[:, 3], tau=six[:, 4], thickness=six[:, 5], rho=None)

    def reg(self) -> torch.Tensor:
        """L2 to the init in raw space (the per-point prior; the grid has TV instead), plus the kNN Laplacian when ``smooth_knn`` > 0."""
        r = ((self.raw_ac - self.raw_ac0) ** 2).mean() + ((self.raw_rough - self.raw_rough0) ** 2).mean()
        if self.smooth_knn > 0:
            for raw in (self.raw_ac, self.raw_rough):
                r = r + self.smooth_knn * ((raw[:, None, :] - raw[self.nbr]) ** 2).mean()
        return r

    def tv(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float64, device=self.raw_ac.device)

    @property
    def voxel_m(self) -> float:
        return 0.0


def knn_indices(xyz: torch.Tensor, k: int = 8, chunk: int = 4096) -> torch.Tensor:
    """(N,k) indices of the k nearest other points (chunked cdist)."""
    out = []
    for lo in range(0, xyz.shape[0], chunk):
        d = torch.cdist(xyz[lo:lo + chunk], xyz); d[torch.arange(d.shape[0]), torch.arange(lo, lo + d.shape[0])] = float("inf")
        out.append(d.topk(k, dim=1, largest=False).indices)
    return torch.cat(out)


class SegmentMaterial(torch.nn.Module):
    """One six-parameter material per scaffold SEGMENT (k-means on position + normal): the room as a handful of materials.
    Same parametrisation as PerPointMaterial, expanded to the points by the segment label."""

    def __init__(self, xyz: torch.Tensor, normals: torch.Tensor | None, segments: int = 16, init_abcd=ITU_GT["concrete"], noise: float = 0.01, iters: int = 30, seed: int = 0):
        super().__init__()
        dev = xyz.device; n = xyz.shape[0]
        nrm = torch.zeros_like(xyz) if normals is None else normals
        feat = torch.cat([xyz / 1.0, 0.5 * nrm], 1)                      # 0.5 m of position ~ one unit of normal
        g = torch.Generator(device=dev).manual_seed(seed)
        centres = feat[torch.randperm(n, generator=g, device=dev)[:segments]].clone()
        for _ in range(iters):
            lab = torch.cdist(feat, centres).argmin(1)
            for j in range(segments):
                sel = lab == j
                if sel.any(): centres[j] = feat[sel].mean(0)
        self.register_buffer("labels", torch.cdist(feat, centres).argmin(1))
        self.inner = PerPointMaterial(centres[:, :3], init_abcd=init_abcd, noise=noise)
        self.segments = segments

    def material_at(self, xyz=None) -> dict:
        m = self.inner.material_at(); lab = self.labels
        return {k: (v[lab] if torch.is_tensor(v) else v) for k, v in m.items()}

    def reg(self): return self.inner.reg()
    def tv(self): return self.inner.tv()
    @property
    def voxel_m(self): return 0.0


def physics_columns(mat: dict, f_ghz: float) -> np.ndarray:
    """(N,6) physics at ``f_ghz`` from a material dict (for saving / figures)."""
    er, ei = eps_from_dispersion(mat["abcd"].detach(), f_ghz)
    n = mat["abcd"].shape[0]
    cols = [er, ei] + [(mat[k].detach() if torch.is_tensor(mat.get(k)) else torch.full((n,), float(OH.SFCW_DEFAULT_ROUGHNESS[k]), device=mat["abcd"].device)) for k in ("sigma_h", "l_c", "tau", "thickness")]
    return torch.stack([c.double() for c in cols], 1).cpu().numpy()


# ----------------------------------------------------------------------------- targets and losses
def load_targets(dev, sfcw_guard: int = GUARD_LO_SFCW, uwb_guard: int | None = None):
    """``sfcw_guard`` / ``uwb_guard``: first supervised bin per band (None = the band's near region, octonet_noise.NEAR_BINS;
    the coupling model lowers it to its adoption bin). The saved masks' wrap guards (last 6 SFCW bins) stay."""
    t = {}
    z = np.load(os.path.join(DATA_OUT, "sfcw_target_S1", "target_pool5.npz"))
    sf = {k: torch.as_tensor(z[k]) for k in z.files}
    sf["rp_mean"] = sf["rp_mean"].to(torch.complex128).to(dev); sf["mask"] = sf["mask"].bool().to(dev)
    sf["mask"][:ON.NEAR_BINS["sfcw"]] = True; sf["mask"][:sfcw_guard] = False
    for k in ("rel", "xcorr", "range_m"):
        if k in sf: sf[k] = sf[k].double().to(dev)
    t["sfcw"] = sf
    z = np.load(os.path.join(DATA_OUT, "uwb_target_S1", "target_pool5.npz"))
    ug = ON.NEAR_BINS["uwb"] if uwb_guard is None else uwb_guard
    um = torch.as_tensor(z["mask"]).bool(); um[:ON.NEAR_BINS["uwb"]] = True; um[:ug] = False
    um[-ON.END_BINS:] = False            # the X4 frame-end artefact (last 0.51 m): a static return peaking at the LAST bin in every recording and scene (2026-10-29 diag), not geometry
    t["uwb"] = dict(env=torch.as_tensor(z["env"], dtype=torch.float64, device=dev), rel=torch.as_tensor(z["rel"], dtype=torch.float64, device=dev).clamp(0, 1),
                    mask=um.to(dev), range_m=torch.as_tensor(z["range_m"], dtype=torch.float64, device=dev), guard_bins=ug)
    t["sfcw"]["guard_bins"] = sfcw_guard
    w = {}
    for k in (1, 2, 3, 4):
        z = np.load(os.path.join(DATA_OUT, "wifi_target_S1", f"target_pool5_rx{k}.npz"))
        w[k] = dict(mean_mag=torch.as_tensor(z["mean_mag"], dtype=torch.float64, device=dev), rel=torch.as_tensor(z["rel"], dtype=torch.float64, device=dev).clamp(0, 1),
                    mask=torch.as_tensor(z["mask"]).bool().to(dev))
    t["wifi"] = w
    return t


def hilbert_env(x: torch.Tensor) -> torch.Tensor:
    X = torch.fft.fft(x); n = x.shape[-1]
    h = torch.zeros(n, dtype=x.dtype, device=x.device); h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    return torch.fft.ifft(X * h).abs()


def sensors_with_gains(sensors: dict, g_tx, g_rx) -> dict:
    """Copy of ``sensors`` whose Vayyar entry carries the linear per-channel gains exp(g_tx), exp(g_rx) (D13)."""
    out = dict(sensors); v = dict(sensors["vayyar"]); v["gain_tx"] = torch.exp(g_tx); v["gain_rx"] = torch.exp(g_rx); out["vayyar"] = v
    return out


def sensors_without_gains(sensors: dict) -> dict:
    out = dict(sensors); out["vayyar"] = {k: v for k, v in sensors["vayyar"].items() if k not in ("gain_tx", "gain_rx")}
    return out


def wifi_sensor():
    return dict(tx=S.WIFI_TX_APPROX, rx={k: S.WIFI_RX_APPROX[k] for k in (1, 2, 3, 4)}, los={1: True, 2: True, 3: True, 4: False})


class BandNuisance:
    """Per-band absolute gain + the ported 4DPS nuisances (octonet_noise): ``log_gain`` (1,) learned, ``floor`` a
    LearnedNoiseFloor or None, ``block`` a CouplingBlock or None. ``None`` for a band = the scale-free loss."""

    def __init__(self, band: str, dev, noise_model: bool, coupling: bool):
        self.band = band
        self.floor = ON.LearnedNoiseFloor.load(band, dev) if noise_model else None
        self.block = ON.CouplingBlock.load(band, dev) if (coupling and ON.coupling_available(band)) else None
        self.log_gain = torch.zeros(1, dtype=torch.float64, device=dev, requires_grad=True)

    def parameters(self):
        return [self.log_gain] + (list(self.floor.parameters()) if self.floor is not None else [])

    @torch.no_grad()
    def init(self, cloud, material, sensors, targets):
        """Iteration-0 gain = median log ratio over the supervised cells; floor starts at the target's 5th-percentile power."""
        # the gain is estimated on the far region only (bins >= NEAR_BINS), where the block (product units) is zero
        if self.band == "sfcw":
            rp = OH.sfcw_head(cloud, material, sensors["vayyar"]); tg = targets["sfcw"]; m = tg["mask"].clone(); m[:ON.NEAR_BINS["sfcw"]] = False
            self.log_gain.copy_(ON.init_log_gain(rp.abs(), tg["rp_mean"].abs(), m))
            if self.floor is not None: self.floor.init_scale(float((tg["rp_mean"].abs()[:, m] ** 2).flatten().quantile(0.05)))
        else:
            rf, _ = OH.uwb_head(cloud, material, sensors["uwb"]); env = OH.hilbert_envelope(rf); tg = targets["uwb"]; m = tg["mask"].clone(); m[:ON.NEAR_BINS["uwb"]] = False
            self.log_gain.copy_(ON.init_log_gain(env, tg["env"], m))
            if self.floor is not None: self.floor.init_scale(float((tg["env"][m] ** 2).quantile(0.05)))

    def state(self) -> dict:
        return dict(log_gain=float(self.log_gain), floor=(self.floor.summary() if self.floor is not None else None),
                    coupling=(dict(adopt_from_bin=self.block.adopt_from_bin, near_bins=self.block.near) if self.block is not None else None))


def smooth_log(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Moving average over the last (range) axis, 'same' size (a geometry-robust view of a log-magnitude profile)."""
    if bins <= 1: return x
    k = torch.ones(1, 1, bins, dtype=x.dtype, device=x.device) / bins
    y = torch.nn.functional.conv1d(x.reshape(-1, 1, x.shape[-1]), k, padding=bins // 2)[..., :x.shape[-1]]
    return y.reshape(x.shape)


def band_loss(band: str, cloud, material, sensors, targets, chunk=4096, nuisance: BandNuisance | None = None, smooth_bins: int = 0):
    """(scalar loss, dict of rendered quantities for figures). With ``nuisance`` the SFCW/UWB render is in absolute units
    (gain, coupling block, floored magnitude) and the loss is un-normalised; without it, median-normalised. ``smooth_bins``
    > 1 compares range-smoothed log magnitudes (SFCW / UWB): the material is then not asked to explain per-bin speckle."""
    if band == "sfcw":
        rp = OH.sfcw_head(cloud, material, sensors["vayyar"], chunk=chunk)
        if nuisance is not None: rp = ON.apply_sfcw(rp, nuisance.log_gain, nuisance.block, nuisance.floor)
        loss, per_bin = ST.sfcw_static_loss(rp, targets["sfcw"], kind="logmag", weight="xcorr", normalize=nuisance is None, smooth_bins=smooth_bins)
        return loss, dict(rp=rp.detach())
    if band == "uwb":
        rf, _ = OH.uwb_head(cloud, material, sensors["uwb"])
        if nuisance is not None: rf, env = ON.apply_uwb(rf, nuisance.log_gain, nuisance.block, nuisance.floor)
        else: env = hilbert_env(rf)
        tg = targets["uwb"]; m = tg["mask"]; w = tg["rel"] * m
        la = smooth_log(torch.log(env + EPS), smooth_bins); lb = smooth_log(torch.log(tg["env"] + EPS), smooth_bins)
        if nuisance is None:
            la = la - la[m].median(); lb = lb - lb[m].median()
        loss = ((la - lb).abs() * w).sum() / (w.sum() + EPS)
        return loss, dict(env=env.detach(), rf=rf.detach())
    if band == "wifi":
        H = OH.wifi_head(cloud, material, sensors["wifi"])                     # (4,2,114)
        tot = 0.0; outs = {}
        for i, k in enumerate((1, 2, 3, 4)):
            tg = targets["wifi"][k]; m = tg["mask"]; w = tg["rel"] * m
            la = torch.log(H[i].abs() + EPS); lb = torch.log(tg["mean_mag"] + EPS)
            la = la - la[:, m].median(dim=1, keepdim=True).values; lb = lb - lb[:, m].median(dim=1, keepdim=True).values
            tot = tot + (((la - lb).abs() * w[None, :]).sum() / (2 * w.sum() + EPS))
            outs[k] = H[i].detach().abs()
        return tot / 4.0, dict(csi=outs)
    raise ValueError(band)


# ----------------------------------------------------------------------------- fit
def setup(dev, n_points, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cloud, material0, sensor_v = render_inputs(SCAFFOLD_DIR, EXTRINSICS, device=dev, n_points=n_points, seed=seed)
    ext = json.load(open(EXTRINSICS))
    sensors = dict(vayyar=sensor_v, uwb=OH.sensor_from_extrinsics(ext, "uwb"), wifi=wifi_sensor())
    return cloud, material0, sensors


def evaluate(bands, cloud, material, sensors, targets, nuis=None):
    with torch.no_grad():
        return {b: band_loss(b, cloud, material, sensors, targets, nuisance=(nuis or {}).get(b))[0].item() for b in bands}


def make_material(kind: str, cloud, voxels: int, dev, free_exponents: bool = False, smooth_knn: float = 0.0, segments: int = 16):
    if kind == "per-point": return PerPointMaterial(cloud["xyz"], free_exponents=free_exponents, smooth_knn=smooth_knn).to(dev)
    if kind == "segment": return SegmentMaterial(cloud["xyz"], cloud.get("normals"), segments=segments).to(dev)
    if kind == "grid": return VoxelGridMaterial(cloud["xyz"], voxels=voxels).to(dev)
    raise ValueError(kind)


def fit(bands, dev, n_points=20000, steps=250, lr=0.03, voxels=28, tv=1e-3, seed=0, tag=None, verbose=True, gains=True, gain_l2=0.01, patience=60,
        material="per-point", l2=0.05, noise_model=True, coupling=True, free_exponents=False, smooth_knn=0.0, segments=16, smooth_loss=(0, 0)):
    """Fit the material on ``bands``. ``material``: 'per-point' (3DPS, default) or 'grid' (T11 voxel field, TV ``tv``);
    ``l2`` = the per-point prior to the init. ``noise_model`` / ``coupling`` (SFCW and UWB): the ported 4DPS nuisances
    (``octonet_noise``), each with a learned absolute gain per band; off = the scale-free median-normalised loss and the
    near-region guard. For the SFCW band, ``gains`` adds the 20 + 20 learnable log-amplitude per-channel gains (D13; L2
    ``gain_l2`` pins their spread). The returned ``sensors['vayyar']`` carries the fitted linear gains; the returned
    ``nuis`` dict (band -> BandNuisance) is needed to render the trained model."""
    cloud, material0, sensors = setup(dev, n_points, seed)
    sensors = sensors_without_gains(sensors)
    targets = load_targets(dev, sfcw_guard=ON.guard_bins("sfcw", coupling), uwb_guard=ON.guard_bins("uwb", coupling))
    grid = make_material(material, cloud, voxels, dev, free_exponents, smooth_knn, segments)
    sm = dict(sfcw=int(smooth_loss[0]), uwb=int(smooth_loss[1]), wifi=0)
    use_gains = bool(gains) and ("sfcw" in bands)
    g_tx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=use_gains)
    g_rx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=use_gains)
    mat0 = dict(abcd=material0["abcd"], rho=None)
    nuis = {}
    for b in ("sfcw", "uwb"):
        if noise_model or coupling:
            nuis[b] = BandNuisance(b, dev, noise_model, coupling); nuis[b].init(cloud, mat0, sensors, targets)
    params = [{"params": list(grid.parameters()), "lr": lr}] + ([{"params": [g_tx, g_rx], "lr": lr}] if use_gains else [])
    nparams = [p for b in bands if b in nuis for p in nuis[b].parameters()]
    if nparams: params.append({"params": nparams, "lr": lr})
    opt = torch.optim.Adam(params)
    # start losses (uniform concrete == the material's init up to noise, gains = 1, iteration-0 gain/floor) on ALL bands for the gate
    start_all = evaluate(BANDS, cloud, mat0, sensors, targets, nuis)
    w = {b: 1.0 / max(start_all[b], 1e-6) for b in bands}
    log = []; t0 = time.time(); best = (float("inf"), None, 0, None)
    best_hist = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        mat = grid.material_at(cloud["xyz"])
        sens = sensors_with_gains(sensors, g_tx, g_rx) if use_gains else sensors
        parts = {b: band_loss(b, cloud, mat, sens, targets, nuisance=nuis.get(b), smooth_bins=sm[b])[0] for b in bands}
        reg_g = gain_l2 * ((g_tx ** 2).mean() + (g_rx ** 2).mean()) if use_gains else 0.0
        reg_m = (l2 * grid.reg()) if isinstance(grid, (PerPointMaterial, SegmentMaterial)) else (tv * grid.tv())
        total = sum(w[b] * parts[b] for b in bands) + reg_m + reg_g
        total.backward(); opt.step()
        rec = dict(step=step, total=float(total), **{b: float(parts[b]) for b in bands}, t=time.time() - t0)
        if use_gains: rec["gain_tx_std"] = float(g_tx.std()); rec["gain_rx_std"] = float(g_rx.std())
        log.append(rec)
        if float(total) < best[0]:
            best = (float(total), {k: v.detach().clone() for k, v in grid.state_dict().items()}, step, (g_tx.detach().clone(), g_rx.detach().clone()),
                    {b: (n.log_gain.detach().clone(), (n.floor.state_dict() if n.floor is not None else None)) for b, n in nuis.items()})
        if verbose and (step % 25 == 0 or step == 1):
            print(f"  [{tag or ','.join(bands)}] step {step:4d} total {float(total):.4f} " + " ".join(f"{b} {float(parts[b]):.4f}" for b in bands)
                  + (f" gains tx/rx std {float(g_tx.std()):.3f}/{float(g_rx.std()):.3f}" if use_gains else "") + f" {time.time()-t0:.0f} s", flush=True)
        # convergence = the BEST total has not improved by 2e-4 for 60 steps (a plain difference test fired on a
        # bouncing loss at ~100 steps and left 0.08 of loss on the table, 2026-10-29); minimum 150 steps
        best_hist.append(min(best_hist[-1], log[-1]["total"]) if best_hist else log[-1]["total"])
        pat = int(patience) if patience is not None else 60
        if pat > 0 and step >= max(150, pat + 1) and best_hist[-1] > best_hist[-(pat + 1)] - 2e-4:
            if verbose: print("  converged", flush=True)
            break
    grid.load_state_dict(best[1])
    if use_gains:
        g_tx, g_rx = best[3]
        sensors = sensors_with_gains(sensors, g_tx, g_rx)                     # the fitted gains ride with the sensors
    for b, (lg, fl) in best[4].items():
        with torch.no_grad(): nuis[b].log_gain.copy_(lg)
        if fl is not None: nuis[b].floor.load_state_dict(fl)
    with torch.no_grad():
        mat = grid.material_at(cloud["xyz"])
        end_all = evaluate(BANDS, cloud, mat, sensors, targets, nuis)
        abcd = mat["abcd"].cpu().numpy()
    res = dict(bands=list(bands), seed=seed, points=int(cloud["xyz"].shape[0]), material=material, voxels=voxels, voxel_m=grid.voxel_m, steps=len(log), best_step=best[2],
               l2=l2, noise_model=bool(noise_model), coupling=bool(coupling), smooth_knn=smooth_knn, segments=(segments if material == "segment" else None), smooth_loss=list(smooth_loss), guards=dict(sfcw=int(targets["sfcw"]["guard_bins"]), uwb=int(targets["uwb"]["guard_bins"])),
               nuisance={b: n.state() for b, n in nuis.items()},
               start=start_all, end=end_all, log=log, runtime_s=time.time() - t0,
               gains=(dict(enabled=True, l2=gain_l2, tx_log=g_tx.detach().cpu().numpy().tolist(), rx_log=g_rx.detach().cpu().numpy().tolist(),
                           tx_std_db=float(20 / math.log(10) * g_tx.detach().std()), rx_std_db=float(20 / math.log(10) * g_rx.detach().std()))
                      if use_gains else dict(enabled=False)))
    res["nuis"] = nuis
    return res, grid, abcd, cloud, sensors, targets


def dispersion_summary(abcd: np.ndarray) -> dict:
    out = {}
    for f in (5.18, 7.0, 65.0):
        er = abcd[:, 0] * f ** abcd[:, 1]; sg = abcd[:, 2] * f ** abcd[:, 3]
        out[f"{f}GHz"] = dict(eps_median=float(np.median(er)), eps_p05=float(np.percentile(er, 5)), eps_p95=float(np.percentile(er, 95)), eps_min=float(er.min()), eps_max=float(er.max()),
                              sigma_median=float(np.median(sg)), sigma_p05=float(np.percentile(sg, 5)), sigma_p95=float(np.percentile(sg, 95)), sigma_min=float(sg.min()), sigma_max=float(sg.max()))
    # nearest ITU class in (log eps', log sigma) at the three frequencies jointly
    names = list(ITU_GT); feats = []
    for nm in names:
        a, b, c, d = ITU_GT[nm]; feats.append(np.concatenate([[np.log(a * f ** b), np.log(c * f ** d)] for f in (5.18, 7.0, 65.0)]))
    feats = np.array(feats)
    pf = np.stack([np.concatenate([[np.log(abcd[i, 0] * f ** abcd[i, 1]), np.log(abcd[i, 2] * f ** abcd[i, 3])] for f in (5.18, 7.0, 65.0)]) for i in range(len(abcd))])
    near = np.argmin(((pf[:, None, :] - feats[None, :, :]) ** 2).sum(-1), axis=1)
    out["nearest_itu_class"] = {nm: int((near == i).sum()) for i, nm in enumerate(names)}
    out["abcd_median"] = np.median(abcd, 0).tolist(); out["abcd_p05"] = np.percentile(abcd, 5, 0).tolist(); out["abcd_p95"] = np.percentile(abcd, 95, 0).tolist()
    return out


def save_run(name, res, grid, abcd, cloud, extra=None):
    os.makedirs(OUT, exist_ok=True)
    res = dict(res); res.pop("nuis", None); res["dispersion"] = dispersion_summary(abcd); res.update(extra or {})
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    g = res.get("gains", {}) or {}
    extra_arrays = (dict(grid=grid.grid.detach().cpu().numpy(), lo=grid.lo.cpu().numpy(), hi=grid.hi.cpu().numpy()) if isinstance(grid, VoxelGridMaterial)
                    else dict(physics_sfcw=physics_columns(grid.material_at(), BAND_FC_GHZ["sfcw"]), physics_uwb=physics_columns(grid.material_at(), BAND_FC_GHZ["uwb"]),
                              physics_columns=np.array(["eps_r", "eps_i", "sigma_h", "l_c", "tau", "thickness"])))
    np.savez(os.path.join(OUT, f"{name}_material.npz"), xyz=cloud["xyz"].cpu().numpy(), abcd=abcd, **extra_arrays,
             gain_tx_log=np.asarray(g.get("tx_log", np.zeros(20)), float), gain_rx_log=np.asarray(g.get("rx_log", np.zeros(20)), float))
    return res


# ----------------------------------------------------------------------------- commands
def cmd_run(a):
    dev = torch.device(a.device); bands = tuple(a.bands.split(","))
    res, grid, abcd, cloud, sensors, targets = fit(bands, dev, a.points, a.steps, a.lr, a.voxels, a.tv, a.seed, tag=a.tag, gains=not a.no_gains, gain_l2=a.gain_l2,
                                                   **fit_kwargs(a))
    nuis = res["nuis"]
    name = a.tag or ("run_" + "_".join(bands) + f"_s{a.seed}")
    sensors0 = sensors_without_gains(sensors)
    # rendered quantities before/after for the figure
    with torch.no_grad():
        mat0 = dict(abcd=torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=dev).expand(cloud["xyz"].shape[0], 4).contiguous(), rho=None)
        mat1 = grid.material_at(cloud["xyz"])
        rend = {}
        for b in BANDS:
            _, r0 = band_loss(b, cloud, mat0, sensors0, targets, nuisance=nuis.get(b)); _, r1 = band_loss(b, cloud, mat1, sensors, targets, nuisance=nuis.get(b))
            if b == "sfcw":
                rend["sfcw_before"] = r0["rp"].abs().mean(0).cpu().numpy(); rend["sfcw_after"] = r1["rp"].abs().mean(0).cpu().numpy()
            elif b == "uwb":
                rend["uwb_before"] = r0["env"].cpu().numpy(); rend["uwb_after"] = r1["env"].cpu().numpy()
            else:
                for k in (1, 2, 3, 4):
                    rend[f"wifi{k}_before"] = r0["csi"][k].cpu().numpy(); rend[f"wifi{k}_after"] = r1["csi"][k].cpu().numpy()
    res = save_run(name, res, grid, abcd, cloud)
    np.savez(os.path.join(OUT, f"{name}_render.npz"), **rend)
    print(json.dumps({k: res[k] for k in ("bands", "start", "end", "best_step", "runtime_s")}, indent=1))
    print("dispersion:", json.dumps({k: v for k, v in res["dispersion"].items() if k in ("65.0GHz", "nearest_itu_class", "abcd_median")}, indent=1))


def cmd_gate(a):
    dev = torch.device(a.device); held = a.held; train = tuple(b for b in BANDS if b != held)
    for seed in [int(s) for s in a.seeds.split(",")]:
        res, grid, abcd, cloud, sensors, targets = fit(train, dev, a.points, a.steps, a.lr, a.voxels, a.tv, seed, tag=f"gate_{held}_s{seed}", gains=not a.no_gains, gain_l2=a.gain_l2, **fit_kwargs(a))
        res = save_run(f"gate_held_{held}_s{seed}", res, grid, abcd, cloud, dict(held=held, held_start=res["start"][held], held_end=res["end"][held]))
        print(f"gate held={held} seed={seed}: uniform concrete {res['held_start']:.4f} -> two-band fit {res['held_end']:.4f}  (train: " + " ".join(f"{b} {res['start'][b]:.3f}->{res['end'][b]:.3f}" for b in train) + ")")


def cmd_report(a):
    import glob
    runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))}
    lines = ["multiband fit report", ""]
    for nm, r in runs.items():
        if nm.startswith("run_"):
            lines.append(f"{nm:28s} start " + " ".join(f"{b} {r['start'][b]:.4f}" for b in BANDS) + " | end " + " ".join(f"{b} {r['end'][b]:.4f}" for b in BANDS) + f" | steps {r['steps']} best {r['best_step']} {r['runtime_s']:.0f}s")
    gate = {}
    for nm, r in runs.items():
        if nm.startswith("gate_held_"):
            gate.setdefault(r["held"], []).append((r["held_start"], r["held_end"]))
    lines.append("")
    for held, vals in gate.items():
        s0 = np.array([v[0] for v in vals]); s1 = np.array([v[1] for v in vals]); gain = s0 - s1
        verdict = "PASS" if (gain.mean() > 0 and gain.mean() > 2 * (gain.std() + 1e-9)) else "FAIL"
        lines.append(f"gate held {held:5s}: uniform concrete {s0.mean():.4f} -> two-band fit {s1.mean():.4f} +- {s1.std():.4f} over {len(vals)} seeds  gain {gain.mean():+.4f} +- {gain.std():.4f}  -> {verdict}")
    txt = "\n".join(lines); print(txt); open(os.path.join(OUT, "report.txt"), "w").write(txt + "\n")
    figure(runs)


def figure(runs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dev = "cpu"; targets = load_targets(dev)
    joint = next((r for n, r in runs.items() if n.startswith("run_sfcw_uwb_wifi")), None)
    jname = next((n for n in runs if n.startswith("run_sfcw_uwb_wifi")), None)
    rend = np.load(os.path.join(OUT, f"{jname}_render.npz")) if jname and os.path.exists(os.path.join(OUT, f"{jname}_render.npz")) else None
    mat = np.load(os.path.join(OUT, f"{jname}_material.npz")) if jname else None
    fig, ax = plt.subplots(3, 4, figsize=(26, 17))
    # (0,0) loss curves
    a = ax[0, 0]
    if joint:
        st = [l["step"] for l in joint["log"]]
        for b in BANDS: a.plot(st, [l[b] / joint["start"][b] for l in joint["log"]], label=f"{b} (rel. start)")
        a.plot(st, [l["total"] for l in joint["log"]], "k", label="total")
    a.set_xlabel("Adam step"); a.set_ylabel("loss / start"); a.set_title("joint fit: per-band loss (normalised) and total"); a.legend(); a.grid(alpha=.3)
    # (0,1) SFCW profile
    a = ax[0, 1]; sf = targets["sfcw"]; m = sf["mask"].numpy(); r = sf["range_m"].numpy()
    def ndb(p):
        d = 20 * np.log10(np.maximum(p, 1e-300)); d = d - np.median(d[m]); d[~m] = np.nan; return d
    a.plot(r, ndb(sf["rp_mean"].abs().mean(0).numpy()), "k", lw=2, label="GT pooled")
    if rend is not None: a.plot(r, ndb(rend["sfcw_before"]), "C0", label="rendered: uniform concrete"); a.plot(r, ndb(rend["sfcw_after"]), "C3", label="rendered: joint fit")
    a.set_xlabel("range (m)"); a.set_ylabel("pair-mean log|rp| dB rel. median"); a.set_title("SFCW (bins < 0.9 m masked: node near-field)"); a.legend(fontsize=8); a.grid(alpha=.3)
    # (0,2) UWB envelope
    a = ax[0, 2]; u = targets["uwb"]; m = u["mask"].numpy(); r = u["range_m"].numpy()
    def udb(p):
        d = 20 * np.log10(np.maximum(p, 1e-300)); d = d - np.median(d[m]); d[~m] = np.nan; return d
    a.plot(r, udb(u["env"].numpy()), "k", lw=1.2, label="GT pooled envelope")
    if rend is not None: a.plot(r, udb(rend["uwb_before"]), "C0", lw=.8, label="rendered: uniform concrete"); a.plot(r, udb(rend["uwb_after"]), "C3", lw=.8, label="rendered: joint fit")
    a.set_xlim(0, 9.9); a.set_xlabel("range (m)"); a.set_ylabel("log env dB rel. median"); a.set_title("UWB (guard 0.29 m)"); a.legend(fontsize=8); a.grid(alpha=.3)
    # (0,3) WiFi
    a = ax[0, 3]
    for i, k in enumerate((1, 2, 3, 4)):
        tg = targets["wifi"][k]; m = tg["mask"].numpy(); sc = np.arange(114)
        g = 20 * np.log10(tg["mean_mag"].numpy()[0]); g -= np.median(g[m]); g[~m] = np.nan
        a.plot(sc, g + 15 * i, "k", lw=1.2, label="GT" if i == 0 else None)
        if rend is not None:
            for key, col, lab in ((f"wifi{k}_before", "C0", "uniform concrete"), (f"wifi{k}_after", "C3", "joint fit")):
                q = 20 * np.log10(rend[key][0] + 1e-300); q -= np.median(q[m]); q[~m] = np.nan; a.plot(sc, q + 15 * i, col, lw=.8, label=lab if i == 0 else None)
        a.text(0, 15 * i + 3, f"Rx-{k}", fontsize=8)
    a.set_xlabel("subcarrier"); a.set_ylabel("log|CSI| dB rel. median (+15 dB per receiver)"); a.set_title("WiFi, antenna 0 (schematic Tx/Rx positions)"); a.legend(fontsize=8); a.grid(alpha=.3)
    # (1,0) per-band alone vs joint bars
    a = ax[1, 0]; names = []; st_v = []; en_alone = []; en_joint = []
    for b in BANDS:
        alone = next((r for n, r in runs.items() if n == f"run_{b}_s0"), None)
        names.append(b); st_v.append(joint["start"][b] if joint else np.nan); en_alone.append(alone["end"][b] if alone else np.nan); en_joint.append(joint["end"][b] if joint else np.nan)
    x = np.arange(3); a.bar(x - 0.25, st_v, 0.25, label="uniform concrete"); a.bar(x, en_alone, 0.25, label="band alone"); a.bar(x + 0.25, en_joint, 0.25, label="joint (3 bands)")
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel("loss"); a.set_title("per-band loss: start / alone / joint"); a.legend(); a.grid(alpha=.3, axis="y")
    # (1,1) gate bars
    a = ax[1, 1]; gate = {}
    for n, r in runs.items():
        if n.startswith("gate_held_"): gate.setdefault(r["held"], []).append((r["held_start"], r["held_end"]))
    if gate:
        hb = list(gate); x = np.arange(len(hb))
        s0 = [np.mean([v[0] for v in gate[h]]) for h in hb]; s1 = [np.mean([v[1] for v in gate[h]]) for h in hb]; e1 = [np.std([v[1] for v in gate[h]]) for h in hb]
        a.bar(x - 0.2, s0, 0.4, label="(a) uniform concrete"); a.bar(x + 0.2, s1, 0.4, yerr=e1, capsize=4, label="(c) fit on the other two bands")
        a.set_xticks(x); a.set_xticklabels([f"held {h}" for h in hb])
    a.set_ylabel("held-band loss"); a.set_title("held-out band gate (3 seeds)"); a.legend(); a.grid(alpha=.3, axis="y")
    # (1,2) eps' map, (1,3) sigma map at 65 GHz
    if mat is not None:
        xyz = mat["xyz"]; ab = mat["abcd"]; er = ab[:, 0] * 65.0 ** ab[:, 1]; sg = ab[:, 2] * 65.0 ** ab[:, 3]
        for a, v, lab in ((ax[1, 2], er, "eps' at 65 GHz"), (ax[1, 3], np.log10(sg), "log10 sigma [S/m] at 65 GHz")):
            sc_ = a.scatter(xyz[:, 0], xyz[:, 1], c=v, s=1.5, cmap="viridis"); plt.colorbar(sc_, ax=a, label=lab)
            a.plot(*S.NODE_POS_S1[1][:2], "r^", ms=9); a.plot(0, 0, "k*", ms=9); a.set_aspect("equal"); a.set_xlabel("X [m]"); a.set_ylabel("Y [m]"); a.set_title(f"fitted {lab} (top-down)")
        # (2,0) eps'(f) and sigma(f) curves
        f = np.logspace(np.log10(2), np.log10(80), 60)
        for a, fn, lab in ((ax[2, 0], lambda ab_, f_: ab_[0] * f_ ** ab_[1], "eps'(f)"), (ax[2, 1], lambda ab_, f_: ab_[2] * f_ ** ab_[3], "sigma(f) [S/m]")):
            for q, col in ((50, "C3"), (5, "C1"), (95, "C1")):
                a.plot(f, [np.percentile([fn(ab[i], ff) for i in range(0, len(ab), max(1, len(ab) // 3000))], q) for ff in f], col, lw=2 if q == 50 else 1, label=f"fitted p{q}" if q != 95 else None)
            for nm, abcd_ in ITU_GT.items(): a.plot(f, [fn(abcd_, ff) for ff in f], "--", lw=.8, label=nm)
            for fc in (5.18, 7.0, 65.0): a.axvline(fc, color="0.6", lw=.6)
            a.set_xscale("log"); a.set_xlabel("frequency [GHz]"); a.set_ylabel(lab); a.set_title(f"{lab}: fitted field (percentiles over points) vs ITU classes"); a.legend(fontsize=7); a.grid(alpha=.3)
            if "sigma" in lab: a.set_yscale("log")
        # (2,2) nearest ITU class histogram
        d = dispersion_summary(ab); a = ax[2, 2]; items = d["nearest_itu_class"]
        a.bar(list(items), list(items.values())); a.set_title("nearest ITU class per point (log eps', log sigma at 5.2/7/65 GHz)"); a.set_ylabel("points"); a.grid(alpha=.3, axis="y")
    # (2,3) text
    a = ax[2, 3]; a.axis("off")
    txt = open(os.path.join(OUT, "report.txt")).read() if os.path.exists(os.path.join(OUT, "report.txt")) else ""
    if mat is not None:
        d = dispersion_summary(mat["abcd"])
        txt += "\n\nfitted field at 65 GHz: eps' median %.2f [%.2f, %.2f] min %.2f max %.2f; sigma median %.3f [%.3f, %.3f] S/m" % (
            d["65.0GHz"]["eps_median"], d["65.0GHz"]["eps_p05"], d["65.0GHz"]["eps_p95"], d["65.0GHz"]["eps_min"], d["65.0GHz"]["eps_max"], d["65.0GHz"]["sigma_median"], d["65.0GHz"]["sigma_p05"], d["65.0GHz"]["sigma_p95"])
        txt += "\nat 5.18 GHz: eps' median %.2f, sigma median %.4f; at 7 GHz: eps' %.2f, sigma %.4f" % (d["5.18GHz"]["eps_median"], d["5.18GHz"]["sigma_median"], d["7.0GHz"]["eps_median"], d["7.0GHz"]["sigma_median"])
    a.text(0, 1, txt, va="top", family="monospace", fontsize=7.5)
    fig.suptitle("Material v2: one ITU-dispersive field on a voxel grid, fitted to SFCW + UWB + WiFi static targets (Scene 1); held-out-band gate", fontsize=13)
    fig.tight_layout()
    viz = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "render_vs_gt"); os.makedirs(viz, exist_ok=True)
    fig.savefig(os.path.join(viz, "multiband_fit.png"), dpi=130); print("figure:", os.path.join(viz, "multiband_fit.png"))


def add_model_args(p):
    """The material / nuisance switches shared by this module's commands and the panels (octonet_eval_panels)."""
    p.add_argument("--material", default="per-point", choices=("per-point", "segment", "grid"), help="per-point = 3DPS (default); segment = one material per k-means segment; grid = the T11 voxel ITU field")
    p.add_argument("--l2", type=float, default=0.05, help="per-point: L2 prior to the init in raw space")
    p.add_argument("--free-exponents", action="store_true", help="per-point: also fit the ITU exponents b, d per point (joint fits only)")
    p.add_argument("--noise-model", default="on", choices=("on", "off"), help="learned receiver noise floor (4DPS model, measured prior: octonet_noise measure)")
    p.add_argument("--coupling", default="on", choices=("on", "off"), help="coherent near-field coupling block instead of the near guard (4DPS model), where adopted")
    p.add_argument("--smooth-knn", type=float, default=0.0, help="per-point: kNN Laplacian prior weight on the raw material (0 = off)")
    p.add_argument("--segments", type=int, default=16, help="segment material: number of k-means segments")
    p.add_argument("--smooth-loss", default="0,0", help="range-smoothing of the log-magnitude loss in bins, 'sfcw,uwb' (0 = per-bin)")


def fit_kwargs(a) -> dict:
    return dict(material=a.material, l2=a.l2, noise_model=a.noise_model == "on", coupling=a.coupling == "on", free_exponents=a.free_exponents,
                smooth_knn=a.smooth_knn, segments=a.segments, smooth_loss=tuple(int(v) for v in a.smooth_loss.split(",")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--points", type=int, default=20000); p.add_argument("--steps", type=int, default=250); p.add_argument("--lr", type=float, default=0.03)
        p.add_argument("--voxels", type=int, default=28); p.add_argument("--tv", type=float, default=1e-3); p.add_argument("--device", default="cuda")
        p.add_argument("--no-gains", action="store_true", help="SFCW: do not fit the 20+20 per-channel gains (pre-D13 behaviour)")
        p.add_argument("--gain-l2", type=float, default=0.01, help="L2 weight on the SFCW log-amplitude channel gains")
        add_model_args(p)
    r = sub.add_parser("run"); common(r); r.add_argument("--bands", default="sfcw,uwb,wifi"); r.add_argument("--seed", type=int, default=0); r.add_argument("--tag", default=None)
    g = sub.add_parser("gate"); common(g); g.add_argument("--held", required=True, choices=BANDS); g.add_argument("--seeds", default="0,1,2")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"run": cmd_run, "gate": cmd_gate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
