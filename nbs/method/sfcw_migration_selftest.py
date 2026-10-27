"""Phase-1 self-test for the XRF55/FMCW -> OctoNet/Vayyar-SFCW renderer migration.

Data-independent checks that run without the OctoNet download. Run in the mmir env:

    python -m method.sfcw_migration_selftest

Covers: (a) Vayyar config numbers vs the datasheet; (b) io_utils SFCW range-res; (c) SFCW
point-target range compression; (d) monostatic-vs-bistatic parity (agree for the compact Vayyar
array, diverge for a WiFi-scale separated pair); (e) FMCW-vs-SFCW range-peak parity on a shared
band; (f) Rasterizer SFCW config parse + FMCW-path regression (unchanged); (g) ra_forward Vayyar
SFCW RA + differentiability. Prints a PASS/FAIL summary and exits non-zero on any failure.
"""
import os, sys, json, tempfile, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_config as C
from preprocessing import vayyar_radar_config as V
from method import freq_material as FM
from method import sfcw_forward as S
from method import ra_forward as RA
from mm3DPS.data.io_utils import compute_range_res_from_cfg

RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def main():
    freqs = C.vayyar_freqs_hz(); dr = C.RANGE_RES_M; bw = C.VAY_BW_HZ

    print("(a) Vayyar config vs datasheet")
    check("range_res 3.75 cm", abs(C.RANGE_RES_M - 0.0375) < 5e-4, f"{C.RANGE_RES_M*100:.3f} cm")
    check("R_max ~3.71 m", abs(C.R_MAX_M - 3.71) < 0.05, f"{C.R_MAX_M:.3f} m")
    check("df ~40.4 MHz", abs(C.VAY_DF_HZ/1e6 - 40.404) < 0.1, f"{C.VAY_DF_HZ/1e6:.3f} MHz")
    check("lambda ~4.61 mm", abs(C.VAY_LAMBDA_M*1000 - 4.612) < 0.02, f"{C.VAY_LAMBDA_M*1000:.3f} mm")
    check("100 freqs, 20x20", len(freqs) == 100 and C.VAY_N_TX == 20 and C.VAY_N_RX == 20)
    cfg = V.build_config()
    check("SFCW config, no chirp keys",
          cfg["modulation_scheme"] == "SFCW" and cfg["geometry_mode"] == "monostatic"
          and not any(k in cfg for k in ("freqSlope", "sampleRate", "numAdcSamples", "rampEndTime")))

    print("(b) io_utils SFCW range-res")
    d = tempfile.mkdtemp(); pv = os.path.join(d, "vayyar.json"); json.dump(cfg, open(pv, "w"))
    check("compute_range_res_from_cfg", abs(compute_range_res_from_cfg(pv) - 0.0375) < 5e-4)

    print("(c) SFCW point-target range compression")
    abcd = torch.tensor([list(FM.ITU_GT["concrete"])], dtype=torch.float64)
    for Rtrue in (0.9, 1.5, 2.7):
        H = S.render_sfcw_mono(abcd, torch.tensor([Rtrue], dtype=torch.float64), freqs)
        peak = int(S.sfcw_to_range(H).abs().argmax())
        check(f"point {Rtrue} m -> bin {peak}", abs(peak - round(Rtrue/dr)) <= 1,
              f"expected ~{round(Rtrue/dr)}")

    print("(d) monostatic-vs-bistatic parity")
    tx = torch.tensor([[-0.02, 0, 0], [0.02, 0, 0]], dtype=torch.float64)     # compact 4 cm
    rx = torch.tensor([[0, 0, -0.02], [0, 0, 0.02]], dtype=torch.float64)
    pos = torch.tensor([[0.3, 1.5, 0.1], [-0.2, 1.2, -0.1]], dtype=torch.float64)
    nrm = torch.tensor([[0, -1.0, 0], [0, -1.0, 0]], dtype=torch.float64)
    ab = torch.tensor([FM.ITU_GT["concrete"], FM.ITU_GT["wood"]], dtype=torch.float64)
    Hm = S.render_sfcw_pairs(ab, pos, tx, rx, freqs, "monostatic", normals=nrm)
    Hb = S.render_sfcw_pairs(ab, pos, tx, rx, freqs, "bistatic", normals=nrm)
    rel_compact = float((Hm - Hb).abs().mean() / Hb.abs().mean())
    wtx = torch.tensor([[-1.5, 0.0, 0.0]], dtype=torch.float64)               # WiFi-scale 3 m sep
    wrx = torch.tensor([[1.5, 3.0, 0.0]], dtype=torch.float64)
    Hm2 = S.render_sfcw_pairs(ab, pos, wtx, wrx, freqs, "monostatic", normals=nrm)
    Hb2 = S.render_sfcw_pairs(ab, pos, wtx, wrx, freqs, "bistatic", normals=nrm)
    rel_wifi = float((Hm2 - Hb2).abs().mean() / Hb2.abs().mean())
    check("compact array: modes agree", rel_compact < 0.05, f"rel diff {rel_compact:.4f}")
    check("WiFi-scale: modes diverge", rel_wifi > 5 * max(rel_compact, 1e-3), f"rel diff {rel_wifi:.3f}")

    print("(e) FMCW-vs-SFCW range-peak parity (shared band, flat material)")
    # flat dispersion (b=d=0) -> material identical across the band; both must peak at the same range
    flat = torch.tensor([[5.0, 0.0, 0.02, 0.0]], dtype=torch.float64)
    Rtrue = 1.8
    # FMCW: carrier 65 GHz over the same 4 GHz band (slope*T_chirp = 4 GHz), K=256 fast-time.
    # Both modulations therefore share range_res = c/(2*B) = dr, so bin*dr -> range for each.
    S_slope, K, chirpT = 4e9 / 60e-6, 256, 60e-6
    adc = FM.render_adc(flat, torch.tensor([Rtrue], dtype=torch.float64), 65.0, S_slope, K, chirpT)
    win = torch.hann_window(K, periodic=False, dtype=torch.float64)
    rp_fmcw = torch.fft.fft(adc.to(torch.complex128) * win.to(torch.complex128)).abs()  # (K,)
    rng_fmcw = int(rp_fmcw[:K // 2].argmax()) * dr         # positive-range half; bin*range_res
    # SFCW over the same band: IFFT-over-freq range compression.
    Hs = S.render_sfcw_mono(flat, torch.tensor([Rtrue], dtype=torch.float64), freqs)
    rng_sfcw = int(S.sfcw_to_range(Hs).abs().argmax()) * dr
    check("SFCW peak range ~ true", abs(rng_sfcw - Rtrue) <= 2 * dr, f"{rng_sfcw:.2f} m vs {Rtrue} m")
    check("FMCW peak range ~ true", abs(rng_fmcw - Rtrue) <= 2 * dr, f"{rng_fmcw:.2f} m vs {Rtrue} m")
    check("FMCW & SFCW peaks agree", abs(rng_fmcw - rng_sfcw) <= 2 * dr,
          f"FMCW {rng_fmcw:.2f} m vs SFCW {rng_sfcw:.2f} m")

    print("(f) Rasterizer SFCW parse + FMCW regression")
    import mm3DPS.rasterizer as R
    import mm3DPS.bsdf_torch as bsdf
    _orig = R.AntennaPatternTorch
    R.AntennaPatternTorch = type("_Stub", (), {"__init__": lambda self, *a, **k: None})
    try:
        r = R.Rasterizer(pv, "/none.ply", "x", "y", device="cpu")
        check("SFCW parse", r.modulation_scheme == "SFCW" and r.K == 100 and r.n_tx == 20
              and abs(r.center_freq - 65e9) < 1 and r.slope == 0.0 and abs(r.range_res - 0.0375) < 5e-4)
        check("SFCW sets bsdf lambda@65GHz", abs(bsdf.WAVELENGTH - FM.C_LIGHT/65e9) < 1e-6)
        bsdf.set_wavelength(3.9e-3)
        fmcw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "3dps_ref", "cascaded_frame_185_aligned_refined.json")
        if os.path.exists(fmcw):
            rf = R.Rasterizer(fmcw, "/none.ply", "x", "y", device="cpu")
            fc = json.load(open(fmcw))
            check("FMCW parse unchanged", rf.modulation_scheme == "FMCW" and rf.freqs is None
                  and abs(rf.slope - float(fc["freqSlope"])) < 1 and rf.num_samples == int(fc["numAdcSamples"]))
            check("FMCW leaves bsdf lambda untouched", abs(bsdf.WAVELENGTH - 3.9e-3) < 1e-9)
    finally:
        R.AntennaPatternTorch = _orig

    print("(g) ra_forward Vayyar SFCW RA + grad")
    pos3 = torch.tensor([[0.0, 1.5, 0.0], [0.3, 2.0, 0.1]], dtype=torch.float64)
    n3 = torch.tensor([[0, -1.0, 0]] * 2, dtype=torch.float64)
    ab3 = torch.tensor([FM.ITU_GT["concrete"], FM.ITU_GT["wood"]], dtype=torch.float64).requires_grad_(True)
    th = torch.tensor([0.1, 0.1], dtype=torch.float64); sh = torch.tensor([1e-4, 1e-4], dtype=torch.float64)
    ra = RA.render_vayyar_ra_disp(ab3, th, sh, pos3, n3, freqs, n_az=64)
    ra.abs().sum().backward()
    check("RA shape (64,100) complex", tuple(ra.shape) == (64, 100) and ra.is_complex())
    check("RA differentiable", torch.isfinite(ab3.grad).all().item() and float(ab3.grad.abs().sum()) > 0)

    n_pass = sum(ok for _, ok in RESULTS); n = len(RESULTS)
    print(f"\nSFCW MIGRATION SELF-TEST: {n_pass}/{n} passed  ->  {'ALL PASS' if n_pass == n else 'FAILURES'}")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
