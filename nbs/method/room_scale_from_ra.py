"""Extract room scale (front-wall distance) from the mmWave RA — a true cross-band geometry cue
(no WiFi used). The radar faces +Y into the room; a flat far wall perpendicular to boresight
retro-reflects, appearing as a far-range return at range = wall distance.

RANGE RESOLUTION is set by the chirp BANDWIDTH: dR = c / (2 B). The XRF55 paper gives the band
(60-64 GHz, TI IWR6843ISK, 256 ADC samples) but NOT the per-chirp swept bandwidth:
  - full 60->64 GHz sweep: B = 4 GHz -> dR = 3.75 cm, max range 9.6 m.
  - config's fitted 0.076 m implies B ~= 1.97 GHz (a ~2 GHz sweep), max range 19.5 m.
So the per-bin localization margin is dR itself: +/-3.75 cm (B=4 GHz) .. +/-7.6 cm (B~2 GHz).
We report the wall in BINS (bandwidth-independent) and back out the dR implied by the physical
WiFi front-wall distance, which discriminates the two bandwidth hypotheses.

Static room = median across clips of the per-clip peak-hold over 17 frames (the moving human
occupies any given (range,az) bin in a minority of clips -> median suppresses it; walls are in
~every clip -> median keeps them). 1/r^4 two-way falloff is compensated to expose the far wall.

  python -m method.room_scale_from_ra --scene Scene1 --nclips 80
"""
import os, sys, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_config as C

C_LIGHT = 299792458.0
N_RANGE_BINS = 256
DR_4GHZ = C_LIGHT / (2 * 4e9)        # 0.0375 m/bin if full 60-64 GHz swept
DR_2GHZ = C_LIGHT / (2 * 1.97e9)     # 0.0761 m/bin (matches the config's fitted value)


def static_ra(scene, nclips):
    files = sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", scene, "mmWave", "*.npy")))
    idx = np.linspace(0, len(files) - 1, min(nclips, len(files))).astype(int)
    stack = []
    for i in idx:
        a = np.load(files[i])[0]                  # (17, 256, 128)
        ra = a[:, :, 64:128]                      # (17, 256, 64) azimuth half
        stack.append(ra.max(0))                   # peak-hold over frames -> (256, 64)
    return np.median(np.stack(stack), 0)          # median across clips -> static (256, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--nclips", type=int, default=80)
    a = ap.parse_args()
    ra = static_ra(a.scene, a.nclips)             # (256 range, 64 azimuth)
    R, A = ra.shape
    bins = np.arange(R)

    # boresight column (sin theta ~ 0 = center azimuth bins)
    cen = A // 2
    bore = ra[:, cen - 2:cen + 3].mean(1)         # (256,) range profile at boresight

    # 1/r^4 compensation to expose far returns (in bin units; scale-free)
    comp = bore * (bins.astype(float) ** 2)
    comp[bins < 7] = 0                             # ignore the near few bins (DC/clutter)
    cn = comp / (comp.max() + 1e-12)

    # WiFi-calibrated front wall (far +Y wall), a PHYSICAL distance from WiFi multipath
    box = np.load(os.path.join(C.NBS_DATA_OUT, a.scene, "wifi_room_box.npy"))
    y1_wifi = box[3]                               # radar at origin -> front-wall distance = y1

    # detect the far wall: farthest prominent peak (>40% of max) that is not the max-range edge
    from scipy.signal import find_peaks
    pk, props = find_peaks(cn, height=0.30, distance=5)
    pk = pk[pk < R - 4]                            # drop the max-range-edge artifact bin
    wall_bin = int(pk[np.argmax(pk)]) if len(pk) else int(np.argmax(cn))
    scene_bin = int(np.argmax(cn))

    print(f"[{a.scene}]  range bins={R}, 256 ADC samples")
    print(f"  dR(B=4GHz)={DR_4GHZ*100:.2f} cm/bin (max {DR_4GHZ*R:.1f} m) | dR(B~2GHz)={DR_2GHZ*100:.2f} cm/bin (max {DR_2GHZ*R:.1f} m)")
    print(f"  strongest scene return : bin {scene_bin:3d}  -> {scene_bin*DR_4GHZ:.2f} m (4GHz) / {scene_bin*DR_2GHZ:.2f} m (2GHz)")
    print(f"  farthest wall peak     : bin {wall_bin:3d}  -> {wall_bin*DR_4GHZ:.2f} m (4GHz) / {wall_bin*DR_2GHZ:.2f} m (2GHz)")
    print(f"  WiFi front wall (phys) : {y1_wifi:.2f} m")
    if wall_bin > 0:
        dr_implied = y1_wifi / wall_bin
        b_implied = C_LIGHT / (2 * dr_implied) / 1e9
        print(f"  -> dR implied by (WiFi wall / mmWave wall bin) = {dr_implied*100:.2f} cm/bin  => B ~= {b_implied:.2f} GHz")
        print(f"     (consistency: matches 4GHz if ~3.75 cm, matches the fitted 0.076 if ~7.6 cm)")
    # bandwidth-independent profile in BINS
    print("  boresight profile (1/r^4-comp, normalized; bins of 8):")
    for b0 in range(0, R, 8):
        seg = cn[b0:b0 + 8]
        print(f"    bin {b0:3d}-{min(b0+8,R-1):3d} | {seg.max():.2f} {'#'*int(seg.max()*40)}")


if __name__ == "__main__":
    main()
