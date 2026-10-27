"""mmWave <-> WiFi band translation via MATERIAL IDENTIFICATION.

Each modality is ~single-band, so a dispersion law cannot be fit from one side. The
well-posed bridge is to identify WHICH materials the surfaces are (a soft assignment over a
known ITU material library) at the MEASURED band, then re-render the OTHER band via those
materials' known frequency response. We render the COMPLEX observable and supervise its
MAGNITUDE (|RA| for mmWave, |CSI| for WiFi) — so XRF55's amplitude-only WiFi is fine.

Baseline ("copy"): recover the per-class complex reflectivity attribute at the source band
and reuse it at the target band (a field attribute with no cross-band model). With one band
it cannot do better than copy -> fails across the 12x mmWave/WiFi frequency gap.

  python -m method.band_translate
"""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm3DPS import bsdf_torch as B
from mm3DPS.data.ra_utils import adc_to_ra_complex_single_chip  # noqa (kept for parity)
from method import ra_forward as RF, wifi_forward as WF
from method.freq_material import eps_from_dispersion, C_LIGHT, ITU_GT

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MMW_GHZ = 60.0
WIFI_GHZ = 5.0
NOISE_DB = -20.0
LIB_NAMES = ["concrete", "glass", "wood", "plasterboard", "brick", "marble"]
LIB = [torch.tensor(ITU_GT[m], dtype=torch.float64, device=DEV) for m in LIB_NAMES]
TRUE_CLASS_MAT = [0, 2, 3]            # concrete, wood, plasterboard for the 3 scene classes


def scene(n=300):
    p = np.load("/home/adnan/Desktop/NBS/data/XRF55/Scene1/scene/pcl.npy")
    idx = np.random.RandomState(1).choice(len(p), n, replace=False)
    p = p[idx]
    pos = torch.tensor(p[:, :3], dtype=torch.float64, device=DEV)
    nrm = torch.tensor(p[:, 3:6], dtype=torch.float64, device=DEV)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = pos.norm(dim=1); keep = (r >= 0.5) & (r < 256 * 0.076)
    pos, nrm = pos[keep], nrm[keep]
    z = pos[:, 2]; q = torch.quantile(z, torch.tensor([1/3, 2/3], device=DEV, dtype=z.dtype))
    cls = (z[:, None] > q[None, :]).sum(1)         # 0..2
    return pos, nrm, cls


def cos_inc(pos, nrm):
    inc = -pos / pos.norm(dim=1, keepdim=True)
    return (inc * nrm).sum(1).clamp(1e-3, 1.0)


def mixture_amp(logits, log_sh, cls, pos, nrm, thick, f_ghz):
    """Per-point complex reflectivity at f from soft material weights over the library."""
    B.set_frequency(f_ghz * 1e9); lam = C_LIGHT / (f_ghz * 1e9)
    ct = cos_inc(pos, nrm); M = pos.shape[0]
    w = torch.softmax(logits, dim=1)[cls]          # (M, Klib)
    R = torch.zeros(M, dtype=torch.complex128, device=DEV)
    for k, abcd in enumerate(LIB):
        er, ei = eps_from_dispersion(abcd, f_ghz)
        Rk, _, _, _ = B.itu_slab_fresnel(er.expand(M).float(), ei.expand(M).float(), ct.float(), thick.float())
        R = R + w[:, k].to(torch.complex128) * Rk.to(torch.complex128)
    sh = torch.exp(log_sh)[cls]
    rough = torch.exp(-0.5 * (4 * math.pi * sh * ct / lam) ** 2)
    return R * rough.to(torch.complex128)


def gt_amp(cls, pos, nrm, thick, f_ghz):
    abcd_pt = torch.stack([LIB[TRUE_CLASS_MAT[int(c)]] for c in cls])
    sh_pt = torch.tensor([[8e-4, 4e-4, 2e-4][int(c)] for c in cls], dtype=torch.float64, device=DEV)
    return RF._amp_from_eps(*eps_from_dispersion(abcd_pt, f_ghz), thick, sh_pt, pos, nrm, f_ghz)


def render_obs(amp, modality, pos, f_ghz, tx=None, rx=None, sc=None, noise=None, g=None):
    if modality == "mmwave":
        return RF._render_ra(amp, pos, f_ghz, 60e12, 256, 60e-6, 64, noise, g, DEV)
    return WF.render_csi(amp, pos, tx, rx, sc, noise_db=noise, gen=g)


def corr(p, gt):
    p, gt = p.abs().flatten(), gt.abs().flatten(); p, gt = p - p.mean(), gt - gt.mean()
    return float((p * gt).sum() / (p.norm() * gt.norm() + 1e-12))


def dbg(p, gt):
    return float(20 * torch.log10((p.abs() - gt.abs()).norm() / (gt.abs().norm() + 1e-12)))


def translate(direction, pos, nrm, cls, thick, wifi):
    src_mod, src_f, tgt_mod, tgt_f = (("mmwave", MMW_GHZ, "wifi", WIFI_GHZ) if direction == "mmwave->wifi"
                                      else ("wifi", WIFI_GHZ, "mmwave", MMW_GHZ))
    def render(amp, mod, f, noise=None, g=None):
        return render_obs(amp, mod, pos, f, wifi["tx"], wifi["rx"], wifi["sc"], noise, g)

    gt_src = render(gt_amp(cls, pos, nrm, thick, src_f), src_mod, src_f, NOISE_DB, torch.Generator(device=DEV).manual_seed(7))
    gt_tgt = render(gt_amp(cls, pos, nrm, thick, tgt_f), tgt_mod, tgt_f)

    # NBS: identify materials at source band, re-render target band
    K = len(cls.unique())
    logits = torch.zeros(K, len(LIB), device=DEV, requires_grad=True)
    log_sh = torch.log(torch.full((K,), 4e-4, device=DEV)).requires_grad_(True)
    opt = torch.optim.Adam([logits, log_sh], lr=0.05)
    for it in range(500):
        opt.zero_grad()
        amp = mixture_amp(logits, log_sh, cls, pos, nrm, thick, src_f)
        pred = render(amp, src_mod, src_f)
        loss = ((pred.abs() - gt_src.abs()) ** 2).sum() / (gt_src.abs() ** 2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            log_sh.clamp_(math.log(1e-5), math.log(5e-3))
    nbs_tgt = render(mixture_amp(logits.detach(), log_sh.detach(), cls, pos, nrm, thick, tgt_f), tgt_mod, tgt_f)

    # baseline: recover per-class complex reflectivity at source, COPY to target
    amp_r = torch.zeros(K, device=DEV, requires_grad=True)
    amp_i = torch.zeros(K, device=DEV, requires_grad=True)
    o = torch.optim.Adam([amp_r, amp_i], lr=0.02)
    for it in range(500):
        o.zero_grad()
        amp = torch.complex(amp_r, amp_i)[cls]
        pred = render(amp.to(torch.complex128), src_mod, src_f)
        loss = ((pred.abs() - gt_src.abs()) ** 2).sum() / (gt_src.abs() ** 2).sum()
        loss.backward(); o.step()
    copy_tgt = render(torch.complex(amp_r.detach(), amp_i.detach())[cls].to(torch.complex128), tgt_mod, tgt_f)

    return ({"direction": direction,
             "nbs_corr": corr(nbs_tgt, gt_tgt), "nbs_db": dbg(nbs_tgt, gt_tgt),
             "copy_corr": corr(copy_tgt, gt_tgt), "copy_db": dbg(copy_tgt, gt_tgt),
             "id_weights": torch.softmax(logits.detach(), 1).cpu().numpy().round(2).tolist()},
            {"gt": gt_tgt, "nbs": nbs_tgt, "copy": copy_tgt, "tgt_mod": tgt_mod, "tgt_f": tgt_f})


def main():
    pos, nrm, cls = scene(300)
    thick = torch.full((pos.shape[0],), 0.15, dtype=torch.float64, device=DEV)
    # WiFi bistatic geometry: Tx and Rx arrays placed in the room
    tx = torch.tensor([[-1.0, 0.5, 0.0], [-1.0, 0.5, 0.03], [-1.0, 0.5, 0.06]], dtype=torch.float64, device=DEV)
    rx = torch.tensor([[4.0, 6.0, 0.0], [4.03, 6.0, 0.0], [4.06, 6.0, 0.0]], dtype=torch.float64, device=DEV)
    wifi = {"tx": tx, "rx": rx, "sc": WF.subcarrier_ghz(WIFI_GHZ, 40.0, 30)}
    print(f"[scene] {pos.shape[0]} pts, classes {[int((cls==k).sum()) for k in range(3)]}, "
          f"true mats {[LIB_NAMES[m] for m in TRUE_CLASS_MAT]}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = os.path.join("output", "band_translate"); os.makedirs(out, exist_ok=True)
    res = {}
    for d in ["mmwave->wifi", "wifi->mmwave"]:
        r, viz = translate(d, pos, nrm, cls, thick, wifi)
        res[d] = r
        print(f"[{d}] NBS corr={r['nbs_corr']:.3f} dB={r['nbs_db']:6.1f} | "
              f"copy corr={r['copy_corr']:.3f} dB={r['copy_db']:6.1f}")
        print(f"        identified material weights/class: {r['id_weights']}  (lib={LIB_NAMES})")
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, (t, key) in zip(axs, [("GT target |%s|" % viz["tgt_mod"], "gt"),
                                      ("NBS synthesized", "nbs"), ("copy baseline", "copy")]):
            im = viz[key].abs().detach().cpu()
            if im.ndim == 2 and im.shape[0] > im.shape[1]:    # RA (range x az)
                ax.imshow(20 * torch.log10(im + 1e-9), aspect="auto", origin="lower")
            else:                                             # CSI (pairs x subcarriers)
                ax.imshow(im, aspect="auto", origin="lower")
            ax.set_title(t)
        fig.suptitle(f"{d}  ({viz['tgt_f']} GHz target)  NBS corr {r['nbs_corr']:.3f} | copy {r['copy_corr']:.3f}")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{d.replace('->','_to_')}.png"), dpi=100); plt.close(fig)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)
    print(f"\nsaved {out}/ (results.json + figures)")


if __name__ == "__main__":
    main()
