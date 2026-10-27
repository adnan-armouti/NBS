"""The 3DPS point-set form: the SMPL-bodied sequence, the fixed global point
set, and per-frame visibility over it.

All three mean SMPL by `DEFAULT_MODEL`, unlike `reconstruction`, which
means the depth model -- co-locating those two senses would silently pick
one of them.
"""

from __future__ import annotations

import hashlib
import json
import matplotlib.pyplot as plt
import shutil
import time
import subprocess
import torch

from mm4DPS import CONFIG_DIR

import argparse

import matplotlib

import numpy as np
from mm4DPS.data import DEFAULT_MODEL, SmplBody
from mm4DPS import HUPR_NEAR_FIELD_M
from pathlib import Path
from mm4DPS import config_centre_m, current_dataset, dispatch


# ----------------------------------------------------------------------
# person_sequence_3dps: Apply the 3DPS point-cloud processing chain to a person sequence.
# ----------------------------------------------------------------------



COS_BORE_MIN = 0.1761  # the mm3DPS default FOV cone

MAX_RANGE_M = 9.4      # HuPR range window upper edge



def fov_mask(points: torch.Tensor, radar_centre: torch.Tensor,
             boresight: torch.Tensor) -> torch.Tensor:
    delta = points - radar_centre
    distance = delta.norm(dim=-1).clamp_min(1e-6)
    cos_bore = ((delta / distance[:, None]) * boresight).sum(-1)
    return ((cos_bore > COS_BORE_MIN)
            & (distance > HUPR_NEAR_FIELD_M) & (distance < MAX_RANGE_M))



def _moeller_trumbore_hit(pts: torch.Tensor, direction: torch.Tensor, dist: torch.Tensor,
                          v0: torch.Tensor, edge1: torch.Tensor, edge2: torch.Tensor,
                          eps: float) -> torch.Tensor:
    """(rays, tris) -> does the open segment point -> radar cross the triangle.

    The triangle operands broadcast: ``(tris, 3)`` for every ray against
    every triangle, ``(rays, tris, 3)`` for a per-ray candidate list.
    """
    origin = pts + eps * direction
    h = torch.cross(direction[:, None, :].expand(-1, v0.shape[-2], -1), edge2, dim=-1)
    a = (edge1 * h).sum(-1)
    parallel = a.abs() < 1e-9
    f = 1.0 / torch.where(parallel, torch.ones_like(a), a)
    s = origin[:, None, :] - v0
    u = f * (s * h).sum(-1)
    q = torch.cross(s, edge1.expand(len(pts), -1, -1), dim=-1)
    v = f * (direction[:, None, :] * q).sum(-1)
    t = f * (edge2 * q).sum(-1)
    return (~parallel & (u >= 0) & (v >= 0) & (u + v <= 1)
            & (t > eps) & (t < dist - 2 * eps))


def _ray_angles(points: torch.Tensor, centre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuth and elevation of the direction radar -> point (the ray's line)."""
    delta = points - centre
    return (torch.atan2(delta[..., 0], delta[..., 1]),
            torch.atan2(delta[..., 2], torch.hypot(delta[..., 0], delta[..., 1])))


class TriangleGrid:
    """The mesh binned by angular cell as seen from the radar: an exact cull.

    Every ray here runs along the line radar -> point, so a triangle can
    only cross it if the triangle covers the ray's direction. The mesh is
    binned by the (azimuth, elevation) box of each triangle's vertices,
    padded one cell either side -- for a great-circle edge's bulge beyond
    its vertices' box (below 1e-5 rad on a body's 1 cm faces) and for the
    float error of the angles -- so a ray asks only the triangles of its
    cell. The hit test itself is unchanged, which is why the result is the
    brute force's, bit for bit: the culled triangles are ones the segment
    cannot touch. 13,776 body faces over a 128 x 128 grid put ~1 face in a
    cell (the padding and the face's dense mesh make it ~400 at the
    fullest), against every face for every ray before.
    """

    def __init__(self, triangles: torch.Tensor, centre: torch.Tensor, cells: int = 128) -> None:
        self.v0 = triangles[:, 0]
        self.edge1 = triangles[:, 1] - self.v0
        self.edge2 = triangles[:, 2] - self.v0
        az, el = _ray_angles(triangles, centre)                         # (T, 3)
        lo = torch.stack([az.amin(1), el.amin(1)], -1)
        hi = torch.stack([az.amax(1), el.amax(1)], -1)
        self.origin = lo.amin(0)
        extent = (hi.amax(0) - self.origin).clamp_min(1e-6)
        self.cells = cells
        self.step = extent / (cells - 2)
        self.wraps = bool((hi[:, 0] - lo[:, 0] > 3.0).any())              # a face across +-pi azimuth
        if self.wraps:
            return
        first = ((lo - self.origin) / self.step).floor().long() - 1 + 1     # padded one cell, shifted by one
        last = ((hi - self.origin) / self.step).floor().long() + 1 + 1
        first = first.clamp(0, cells - 1)
        last = last.clamp(0, cells - 1)
        span = last - first + 1                                            # (T, 2)
        per_triangle = (span[:, 0] * span[:, 1])
        tri = torch.repeat_interleave(torch.arange(len(triangles), device=triangles.device), per_triangle)
        offsets = torch.cumsum(per_triangle, 0) - per_triangle
        local = torch.arange(int(per_triangle.sum()), device=triangles.device) - offsets[tri]
        width = span[tri, 1]
        cell_x = first[tri, 0] + local // width
        cell_y = first[tri, 1] + local % width
        cell = cell_x * cells + cell_y
        order = torch.argsort(cell)
        cell, tri = cell[order], tri[order]
        counts = torch.bincount(cell, minlength=cells * cells)
        self.max_per_cell = int(counts.max())
        starts = torch.cumsum(counts, 0) - counts
        # (cells^2, M) triangle ids, -1 where a cell has fewer than M
        self.table = torch.full((cells * cells, self.max_per_cell), -1, dtype=torch.long,
                                device=triangles.device)
        slot = torch.arange(len(cell), device=cell.device) - starts[cell]
        self.table[cell, slot] = tri

    def on_grid(self, points: torch.Tensor, centre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Which rays fall on the grid at all, and their cells.

        A ray off the grid points where no triangle does: nothing to test,
        visible.
        """
        az, el = _ray_angles(points, centre)
        index = ((torch.stack([az, el], -1) - self.origin) / self.step).floor().long() + 1
        inside = ((index >= 0) & (index < self.cells)).all(-1)
        return inside, (index[:, 0] * self.cells + index[:, 1]).clamp(0, self.cells * self.cells - 1)


def ray_cast_visible(points: torch.Tensor, triangles: torch.Tensor,
                     radar_centre: torch.Tensor, eps: float = 1e-3,
                     chunk: int = 4096, grid: "TriangleGrid | None" = None) -> torch.Tensor:
    """Moeller-Trumbore segment test: point -> radar centre vs the mesh.

    A point is occluded when any triangle intersects the open segment
    between the point (nudged eps toward the radar, past its own faces)
    and the radar. Mirrors mm3DPS _ray_test_visibility_batched. Each ray is
    tested against the triangles of its angular cell (`TriangleGrid`, built
    here when not given), which is the whole mesh's answer exactly; the
    all-pairs form runs only for a mesh the grid cannot bin.
    """
    visible = torch.ones(len(points), dtype=torch.bool, device=points.device)
    if len(triangles) == 0 or len(points) == 0:
        return visible
    if grid is None:
        grid = TriangleGrid(triangles, radar_centre)
    if grid.wraps:
        asked = torch.arange(len(points), device=points.device)
    else:
        inside, cell = grid.on_grid(points, radar_centre)
        asked = inside.nonzero(as_tuple=True)[0]
        cell = cell[asked]
    for start in range(0, len(asked), chunk):
        rows = asked[start:start + chunk]
        pts = points[rows]
        to_radar = radar_centre - pts
        dist = to_radar.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = to_radar / dist
        if grid.wraps:
            hit = _moeller_trumbore_hit(pts, direction, dist, grid.v0[None], grid.edge1[None],
                                        grid.edge2[None], eps)
        else:
            ids = grid.table[cell[start:start + chunk]]                   # (rays, M)
            safe = ids.clamp_min(0)
            hit = _moeller_trumbore_hit(pts, direction, dist, grid.v0[safe], grid.edge1[safe],
                                        grid.edge2[safe], eps) & (ids >= 0)
        visible[rows] = ~hit.any(dim=1)
    return visible















# ----------------------------------------------------------------------
# global_pointset: Build the fixed global scene point set for per-frame visibility rendering.
# ----------------------------------------------------------------------
matplotlib.use("Agg")









def radar_centre_and_boresight(config_dir: Path, device: torch.device):
    """The rig's base pose: the mean of the dataset's board centres, looking down +Y."""
    dataset = current_dataset()
    centres = np.stack([
        config_centre_m(json.loads(dataset.config_path(radar, config_dir).read_text()))
        for radar in dataset.radars
    ])
    centre = torch.as_tensor(centres.mean(0), dtype=torch.float32, device=device)
    boresight = torch.tensor([0.0, 1.0, 0.0], device=device)
    return centre, boresight












# ----------------------------------------------------------------------
# compute_frame_visibility: Per-frame visibility of the global point set (scene + person).
# ----------------------------------------------------------------------
matplotlib.use("Agg")





def rays_near_sphere(points: torch.Tensor, target: torch.Tensor,
                     centre: torch.Tensor, radius: float) -> torch.Tensor:
    """Points whose segment to the radar passes within radius of target.

    The person occupies a small solid angle; only scene points behind that
    cone can possibly be shadowed, so everything else skips the ray cast.
    """
    to_radar = centre - points
    length = to_radar.norm(dim=-1).clamp_min(1e-6)
    direction = to_radar / length[:, None]
    to_target = target - points
    along = (to_target * direction).sum(-1)
    closest = points + along.clamp(min=0.0)[:, None] * direction
    within_segment = (along > 0) & (along < length)
    return within_segment & ((closest - target).norm(dim=-1) < radius)



def frame_visibility(person_pcl: torch.Tensor, scene_points: torch.Tensor,
                     scaffold_triangles: torch.Tensor, faces: torch.Tensor,
                     centre: torch.Tensor, boresight: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    person_points = person_pcl[:, :3]
    person_triangles = person_points[faces]
    grid = TriangleGrid(person_triangles, centre)

    # (a) person: FOV + self-occlusion, then (b) the scaffold, cone-culled.
    person_visible = fov_mask(person_points, centre, boresight)
    person_visible &= ray_cast_visible(person_points, person_triangles, centre, grid=grid)
    body_centre = person_points.mean(0)
    body_radius = (person_points - body_centre).norm(dim=-1).max() + 0.05
    if len(scaffold_triangles):
        tri_centroids = scaffold_triangles.mean(dim=1)
        blocking = rays_near_sphere(
            # a scaffold triangle can only occlude the person if it sits on the
            # radar side of the body; test its centroid as a pseudo-point
            tri_centroids, body_centre, centre,
            radius=body_radius + scaffold_triangles.flatten(0, 1).std(),
        ) & (tri_centroids.norm(dim=-1) < (body_centre - centre).norm() + body_radius)
        person_visible &= ray_cast_visible(person_points, scaffold_triangles[blocking], centre)

    # (c) scene points: only those whose ray grazes the person's sphere
    # (the grid answers "none" for the rest at the cost of their angles).
    candidates = rays_near_sphere(scene_points, body_centre, centre, body_radius).nonzero(as_tuple=True)[0]
    scene_visible = torch.ones(len(scene_points), dtype=torch.bool,
                               device=scene_points.device)
    scene_visible[candidates] = ray_cast_visible(
        scene_points[candidates], person_triangles, centre, grid=grid)
    return person_visible, scene_visible


def sequence_visibility(person: torch.Tensor, scene_points: torch.Tensor,
                        scaffold_triangles: torch.Tensor, faces: torch.Tensor,
                        centre: torch.Tensor, boresight: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """`frame_visibility` over a whole sequence: ``(frames, V)`` and ``(frames, N)`` bool, on the device."""
    person_visible = torch.zeros(person.shape[:2], dtype=torch.bool, device=person.device)
    scene_visible = torch.zeros((len(person), len(scene_points)), dtype=torch.bool, device=person.device)
    for frame in range(len(person)):
        person_visible[frame], scene_visible[frame] = frame_visibility(
            person[frame], scene_points, scaffold_triangles, faces, centre, boresight)
    return person_visible, scene_visible



def draw_panel(axis, scene: np.ndarray, person: np.ndarray, title: str,
               shadowed: np.ndarray | None = None, view: str = "front") -> None:
    if view == "front":
        xi, yi = 0, 2
        scene_order = np.argsort(-scene[:, 1])
        person_order = np.argsort(-person[:, 1])
        colour_axis, xlabel, ylabel = 2, "x (m)", "height z (m)"
        xlim, ylim = (-2.6, 2.6), (-1.0, 2.3)
    else:  # top view: highest surfaces painted last, as seen from above
        xi, yi = 0, 1
        scene_order = np.argsort(scene[:, 2])
        person_order = np.argsort(person[:, 2])
        colour_axis, xlabel, ylabel = 2, "x (m)", "range y (m)"
        xlim, ylim = (-2.6, 2.6), (0.0, 7.3)
    axis.scatter(scene[scene_order, xi], scene[scene_order, yi], s=0.8,
                 color="#9a9a9a", alpha=0.55)
    if shadowed is not None and len(shadowed):
        axis.scatter(shadowed[:, xi], shadowed[:, yi], s=2.4, color="#d62728",
                     label="scene points in the person's shadow")
        axis.legend(fontsize=8, loc="upper right")
    axis.scatter(person[person_order, xi], person[person_order, yi], s=1.6,
                 c=person[person_order, colour_axis], cmap="plasma")
    if view == "top":
        axis.scatter([0.0], [0.0], marker="^", s=60, color="#ff2d55")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal")
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.2)



def run_compute_frame_visibility(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_np = np.load(args.global_dir / "global_scene_points.npy")
    scene_points = torch.as_tensor(scene_np[:, :3], device=device)
    triangles_path = args.global_dir / "scaffold_triangles.npy"
    # The LiDAR scene has points but no mesh: with no triangles the person
    # keeps its FOV and self-occlusion terms and the scene cannot occlude it.
    scaffold_triangles = torch.as_tensor(
        np.load(triangles_path) if triangles_path.is_file()
        else np.zeros((0, 3, 3), dtype=np.float64), device=device)
    centre, boresight = radar_centre_and_boresight(args.config_dir, device)

    body = SmplBody(args.model_path)
    faces = body.faces.to(device=device, dtype=torch.long)

    metadata = json.loads((args.sequence_dir / "metadata.json").read_text())
    frames = int(metadata["frames"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "person_frames").mkdir(exist_ok=True)

    scratch = args.output_dir / ".visibility_frames"
    if args.video:
        scratch.mkdir(exist_ok=True)
    payloads = [dict(np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz"))
                for frame in range(frames)]
    if any(len(payload["pcl"]) != body.vertex_count for payload in payloads):
        raise ValueError("per-frame visibility needs the stride-1 export")
    person = torch.as_tensor(np.stack([payload["pcl"][:, :3] for payload in payloads]), device=device)
    person[:, :, 1] += args.translation_y_m            # into the renderer frame the scene lives in
    person_visible, scene_visible = sequence_visibility(person, scene_points, scaffold_triangles, faces,
                                                        centre, boresight)
    person_visible, scene_visibility = person_visible.cpu().numpy(), scene_visible.cpu().numpy()
    person_counts = person_visible.sum(1).tolist()
    shadow_counts = (~scene_visibility).sum(1).tolist()
    for frame, payload in enumerate(payloads):
        payload["visible"] = person_visible[frame]
        np.savez(args.output_dir / "person_frames" / f"frame_{frame:03d}.npz", **payload)
        if args.video:
            person_np = person[frame].cpu().numpy()
            figure, axes = plt.subplots(2, 2, figsize=(14.5, 11.5), dpi=110)
            global_title = (f"global point set: {len(scene_np)} scene + "
                            f"{body.vertex_count} person = "
                            f"{len(scene_np) + body.vertex_count}")
            visible_title = (f"visible this frame: "
                             f"{int(scene_visibility[frame].sum())} scene + "
                             f"{person_counts[frame]} person")
            for row, view in enumerate(("front", "top")):
                draw_panel(axes[row][0], scene_np, person_np[:, :3],
                           f"{global_title}" if row == 0 else "top view: global",
                           view=view)
                draw_panel(axes[row][1], scene_np[scene_visibility[frame]],
                           person_np[payload['visible']][:, :3],
                           f"{visible_title}" if row == 0 else "top view: visible",
                           shadowed=scene_np[~scene_visibility[frame]],
                           view=view)
            figure.suptitle(f"frame {frame:3d}/{frames}   "
                            f"(fixed N, per-frame binary visibility)", fontsize=11)
            figure.savefig(scratch / f"{frame:05d}.png", bbox_inches="tight",
                           facecolor="white")
            plt.close(figure)

    np.save(args.output_dir / "scene_visibility.npy", scene_visibility)
    if args.global_dir.resolve() != args.output_dir.resolve():
        shutil.copy(args.global_dir / "global_scene_points.npy",
                    args.output_dir / "global_scene_points.npy")

    metadata["frame_visibility"] = {
        "global_scene_points": int(len(scene_np)),
        "person_points": int(body.vertex_count),
        "global_total": int(len(scene_np) + body.vertex_count),
        "person_visible": {
            "median": int(np.median(person_counts)),
            "min": int(min(person_counts)), "max": int(max(person_counts))},
        "scene_shadowed_by_person": {
            "median": int(np.median(shadow_counts)),
            "min": int(min(shadow_counts)), "max": int(max(shadow_counts))},
        "steps_per_frame": ["fov", "ray_cast_occlusion"],
        "note": "no cosine, no FPS per frame: visibility is deterministic "
                "and identity-preserving; beam gain lives in the renderer",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.output_dir.resolve() != args.sequence_dir.resolve():   # in place is allowed
        for candidate in sorted(args.sequence_dir.glob("*")):
            if candidate.is_file() and candidate.name != "metadata.json":
                shutil.copy(candidate, args.output_dir / candidate.name)

    if args.video:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
             "-i", str(scratch / "%05d.png"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             str(args.video)], check=True)
        for path in scratch.glob("*.png"):
            path.unlink()
        scratch.rmdir()
    return {"status": "completed", "frames": frames,
            "visibility": metadata["frame_visibility"],
            "output_dir": str(args.output_dir)}



def make_parser_compute_frame_visibility() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True,
                        help="stride-1, radar-calibrated person sequence")
    parser.add_argument("--global-dir", type=Path, required=True,
                        help="output of mm4DPS.preprocessing.global_pointset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-y-m", type=float, required=True)
    parser.add_argument("--video", type=Path,
                        help="write the two-panel visibility video here")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    return parser


def _entry_compute_frame_visibility():
        print(json.dumps(run_compute_frame_visibility(make_parser_compute_frame_visibility().parse_args()), indent=2))



# ----------------------------------------------------------------------
# scene_shadow_mask: Per-frame shadowing of the RENDERER's background points by the person.
# ----------------------------------------------------------------------
# `compute_frame_visibility` above masks `global_scene_points.npy`, an
# 9k-12k-point set built by `global_pointset`. The renderer and the velocity
# trainer do NOT use that set: their background is the 20000-point 3DPS model
# in the checkpoint's `shared_background_state`, sampled from the vendored
# `hupr_multiview` scene with `enable_occlusion=False`. The two index spaces
# are unrelated, so `scene_visibility.npy` cannot be applied to a render.
# This command computes the same quantity over the checkpoint's own points.




def load_person_sequence(sequence_dir: Path, frames: int, vertex_count: int,
                         translation_y_m: float, device) -> torch.Tensor:
    """Every frame's SMPL vertices as one (frames, V, 3) tensor on the device.

    The per-frame `np.load` this replaces did 600 archive opens inside the
    timed loop, which is I/O the GPU then waited on. Reading once and moving
    once turns 600 host round trips into one.
    """
    stacked = np.empty((frames, vertex_count, 3), dtype=np.float32)
    for frame in range(frames):
        payload = np.load(sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")
        pcl = payload["pcl"]
        if len(pcl) != vertex_count:
            raise ValueError(
                "scene shadowing needs the SMPL person export "
                f"({vertex_count} points), got {len(pcl)} at frame {frame}")
        stacked[frame] = pcl[:, :3]
    person = torch.from_numpy(stacked).to(device, non_blocking=True)
    person[:, :, 1] += translation_y_m          # into the renderer's frame
    return person


def silhouette_candidates(background: torch.Tensor, person: torch.Tensor,
                          centre: torch.Tensor) -> torch.Tensor:
    """Scene points that could possibly be shadowed, by ANGULAR extent.

    `rays_near_sphere` bounds the body with a sphere, and a standing person's
    sphere has a ~1 m radius. Projected from the radar to the back wall that is
    a 2.2 m wide cone, so the cull kept 76% of the cloud (measured) and the ray
    cast did 13380 x 13776 tests per frame for nothing.

    A person is NARROW in azimuth and TALL in elevation; a sphere conflates the
    two and pays the tall axis on both. This bounds the body by its actual
    angular box as seen from the radar, which is the silhouette's own bounding
    box, so it is conservative -- no point that the mesh could occlude is
    dropped -- and it is exact in the sense that matters: the ray cast decides,
    this only chooses whom to ask.

    Returns (frames, N) bool.
    """
    def angles(points):
        delta = points - centre
        distance = delta.norm(dim=-1).clamp_min(1e-6)
        return (torch.atan2(delta[..., 0], delta[..., 1]),      # azimuth
                torch.asin((delta[..., 2] / distance).clamp(-1.0, 1.0)),
                distance)

    scene_az, scene_el, scene_r = angles(background)             # (N,)
    body_az, body_el, body_r = angles(person)                    # (F, V)
    # a hair of slack so a grazing ray is asked rather than assumed clear
    pad = 1e-3
    inside = ((scene_az[None] >= body_az.amin(1, keepdim=True) - pad)
              & (scene_az[None] <= body_az.amax(1, keepdim=True) + pad)
              & (scene_el[None] >= body_el.amin(1, keepdim=True) - pad)
              & (scene_el[None] <= body_el.amax(1, keepdim=True) + pad))
    return inside & (scene_r[None] > body_r.amin(1, keepdim=True))


def scene_shadow_sequence(background: torch.Tensor, person: torch.Tensor,
                          faces: torch.Tensor, centre: torch.Tensor,
                          chunk_frames: int = 8) -> torch.Tensor:
    """Per-frame visibility for a whole sequence, without leaving the device.

    Returns (frames, N) bool, True = visible. Same cone-cull-then-
    Moeller-Trumbore as `scene_shadow`, and it is gated against that function
    point for point; what changes is only where the work happens.

    Three things the per-frame version did that cost more than the geometry:
      - `bool(candidates.any())` synchronised the device every frame,
      - `.cpu().numpy()` copied every frame's row back separately,
      - the mask lived in numpy, so the result was assembled on the host.
    Here the whole mask is one device tensor, the cone cull is batched across a
    chunk of frames at once, and the only transfer is the finished array.

    The ray cast still runs per frame, because its candidate count varies and
    padding every frame to the worst case would do more work than it saves.
    What made it expensive was not the cast but the CULL: see
    `silhouette_candidates`.
    """
    frames, _, _ = person.shape
    visible = torch.ones((frames, len(background)), dtype=torch.bool,
                         device=background.device)
    for start in range(0, frames, chunk_frames):
        block = person[start:start + chunk_frames]
        near = silhouette_candidates(background, block, centre)        # (B, N)
        for offset in range(len(block)):
            candidates = near[offset]
            index = candidates.nonzero(as_tuple=True)[0]
            if index.numel() == 0:
                continue
            visible[start + offset, index] = ray_cast_visible(
                background.index_select(0, index), block[offset][faces], centre)
    return visible


def run_scene_shadow_mask(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    background = torch.as_tensor(
        checkpoint["shared_background_state"]["positions"],
        dtype=torch.float32, device=device)
    centre, _ = radar_centre_and_boresight(args.config_dir, device)

    body = SmplBody(args.model_path)
    faces = body.faces.to(device=device, dtype=torch.long)
    frames = int(json.loads(
        (args.sequence_dir / "metadata.json").read_text())["frames"])

    started = time.perf_counter()
    person = load_person_sequence(args.sequence_dir, frames, body.vertex_count,
                                  args.translation_y_m, device)
    loaded = time.perf_counter()
    visible = scene_shadow_sequence(background, person, faces, centre,
                                    chunk_frames=args.chunk_frames)
    if device.type == "cuda":
        torch.cuda.synchronize()
    computed = time.perf_counter()

    mask = visible.cpu().numpy()                 # ONE transfer, at the end
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask)
    # BIND THE MASK TO THE GEOMETRY IT WAS COMPUTED ON.
    #
    # Point count cannot do this: the background model is downsampled to 20000
    # for every sequence, so seq102's and seq256's masks are both (600, 20000)
    # and a count check passes on the wrong one. After R083 the per-sequence
    # alignments differ by up to 1.6 m, so using the wrong mask is not a
    # near-miss -- it shadows the wrong points entirely, and silently.
    #
    # The checksum is over the exact float32 positions the mask was cast
    # against, so it distinguishes not just the sequence but the ALIGNMENT:
    # re-fit the transform and the old mask stops validating, which is what
    # should happen.
    digest = hashlib.sha256(
        background.detach().cpu().numpy().astype(np.float32).tobytes()
    ).hexdigest()
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "mask": args.output.name,
        "sequence": args.sequence,
        "point_checksum_sha256": digest,
        "background_points": int(len(background)),
        "frames": frames,
        "checkpoint": str(args.checkpoint),
        "translation_y_m": args.translation_y_m,
        "binds": "the sha256 is of the float32 scene positions this mask was "
                 "cast against. Point count cannot identify a mask -- every "
                 "sequence downsamples to the same 20000 -- so a consumer must "
                 "check this, not the shape.",
    }, indent=2) + "\n")
    shadowed = (~mask).sum(1)
    return {
        "output": str(args.output),
        "sidecar": str(args.output.with_suffix(".json")),
        "sequence": args.sequence,
        "point_checksum_sha256": digest,
        "frames": frames,
        "background_points": int(len(background)),
        "seconds": {"load_person": round(loaded - started, 2),
                    "visibility": round(computed - loaded, 2),
                    "per_frame_ms": round(1000 * (computed - loaded) / frames, 2)},
        "shadowed_per_frame": {
            "min": int(shadowed.min()), "median": int(np.median(shadowed)),
            "max": int(shadowed.max()), "mean": round(float(shadowed.mean()), 1)},
        "points_ever_shadowed": int((~mask).any(0).sum()),
        "points_always_shadowed": int((~mask).all(0).sum()),
    }


def make_parser_scene_shadow_mask() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="person checkpoint holding shared_background_state")
    parser.add_argument("--sequence-dir", type=Path, required=True,
                        help="SMPL person sequence (6890 points per frame)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-y-m", type=float, required=True)
    parser.add_argument("--sequence", type=int, required=True,
                        help="recorded in the sidecar so a consumer can tell "
                             "which sequence this mask belongs to; the point "
                             "count cannot, they are all 20000")
    parser.add_argument("--chunk-frames", type=int, default=8,
                        help="frames whose cone cull is batched together; the "
                             "cull is (frames x points) so this trades memory "
                             "for launches")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config-dir", type=Path,
                        default=CONFIG_DIR)
    return parser


def _entry_scene_shadow_mask():
        print(json.dumps(run_scene_shadow_mask(make_parser_scene_shadow_mask().parse_args()), indent=2))




COMMANDS = {
    "compute-frame-visibility": _entry_compute_frame_visibility,
    "scene-shadow-mask": _entry_scene_shadow_mask,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
