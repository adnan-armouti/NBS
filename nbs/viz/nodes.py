"""viz/nodes: OctoNet node layout (2D / 3D) + per-node sensor availability matrix.

Run from nbs/:  python -m viz.nodes   -> ../viz/nodes/{nodes_topdown_2d,nodes_3d,node_sensor_matrix}.png + captions.txt
Positions: octonet_scene (Node1-5 = MoCap rigid bodies, Scene 1; WiFi/obstacle = paper Fig. 2a schematic).
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle, Polygon
from matplotlib.lines import Line2D

from preprocessing import octonet_scene as S, octonet_io as IO, octonet_config as C

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "nodes")
os.makedirs(OUT, exist_ok=True)

NODE_COL = {1: "tab:red", 2: "tab:orange", 3: "tab:green", 4: "tab:purple", 5: "tab:brown"}
CAM_NODES = (1, 3, 5)
VAYYAR_RMAX = 3.71
HALF = S.SENSING_AREA_M / 2

# 20-joint order of mocap_pose/*_skeleton.npy, INFERRED from joint heights (OptiTrack 21-bone skeleton
# without 'Head'): Hip Ab Chest Neck LShoulder LUArm LFArm LHand RShoulder RUArm RFArm RHand
# LThigh LShin LFoot LToe RThigh RShin RFoot RToe.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10), (10, 11),
         (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]


def heading_deg(p, target=(0.0, 0.0)):
    return np.degrees(np.arctan2(target[1] - p[1], target[0] - p[0]))


def obstacle_poly():
    c = S.OBSTACLE_APPROX["center"][:2]; L = S.OBSTACLE_APPROX["length_m"]; T = max(S.OBSTACLE_APPROX["thickness_m"], 0.06)
    # board oriented along the (schematic) top edge of the square, i.e. along MoCap Y
    return np.array([[c[0] - T / 2, c[1] - L / 2], [c[0] + T / 2, c[1] - L / 2], [c[0] + T / 2, c[1] + L / 2], [c[0] - T / 2, c[1] + L / 2]])


def load_skeleton_frame():
    """One standing-ish frame (max neck height) of a Scene-1 MoCap clip, metres, Z up."""
    files = sorted(glob.glob(os.path.join(C.OCTONET_ROOT, "mocap_pose", "*_skeleton.npy")))
    pick = [f for f in files if "subject_2" in f and "sit" in f] or files
    pos, _ = IO.load_mocap(pick[0])
    i = int(np.argmax(pos[:, 3, 2]))
    return pos[i], os.path.basename(pick[0])


def fig_topdown():
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fc="#e8f3e2", ec="k", lw=1.2, ls="--", zorder=0))
    ax.text(-HALF + 0.1, -HALF + 0.55, "sensing area 4.1 m x 4.1 m\n(centred on the MoCap origin, assumed)", fontsize=8, va="bottom")
    # camera FoV wedges
    for n in CAM_NODES:
        p = S.NODE_POS_S1[n]; h = heading_deg(p)
        ax.add_patch(Wedge(p[:2], 4.0, h - S.D455_HFOV_DEG / 2, h + S.D455_HFOV_DEG / 2, fc=NODE_COL[n], alpha=0.10, ec=NODE_COL[n], lw=0.6, zorder=1))
    # Vayyar boresight + R_max from node_1
    p1 = S.NODE_POS_S1[1]; h1 = heading_deg(p1)
    ax.add_patch(Wedge(p1[:2], VAYYAR_RMAX, h1 - 60, h1 + 60, fc="none", ec="tab:red", lw=1.4, ls=":", zorder=2))
    ax.annotate("", xy=(p1[0] + 1.5 * np.cos(np.radians(h1)), p1[1] + 1.5 * np.sin(np.radians(h1))), xytext=p1[:2],
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.4), zorder=3)
    ax.text(-3.5, 2.7, "Vayyar SFCW boresight\nR_max = 3.71 m (dotted arc)\nUWB X4M200: R_max 9.9 m (off-plot)", color="tab:red", fontsize=8)
    # nodes
    for n, p in S.NODE_POS_S1.items():
        ax.add_patch(Circle(p[:2], 0.17, fc=NODE_COL[n], ec="k", lw=1.0, zorder=5))
        ax.text(p[0], p[1], str(n), ha="center", va="center", color="w", fontsize=10, weight="bold", zorder=6)
        dx, dy = {1: (0.25, 0.25), 2: (0.25, 0.25), 3: (0.25, 0.1), 4: (-1.15, 0.05), 5: (-0.3, -0.75)}[n]
        ax.text(p[0] + dx, p[1] + dy, f"Node {n}\n({p[0]:+.2f}, {p[1]:+.2f}) m\nz = {p[2]:.2f} m", fontsize=7.5, zorder=6)
    # WiFi (schematic)
    tx = S.WIFI_TX_APPROX
    ax.add_patch(Polygon([[tx[0] + 0.2, tx[1]], [tx[0] - 0.15, tx[1] + 0.17], [tx[0] - 0.15, tx[1] - 0.17]], fc="none", ec="tab:red", hatch="///", lw=1.2, zorder=5))
    ax.text(tx[0] + 0.25, tx[1] - 0.45, "WiFi Tx (approx.)", fontsize=7.5, color="tab:red")
    for k, r in S.WIFI_RX_APPROX.items():
        ax.add_patch(Polygon([[r[0] + 0.2, r[1]], [r[0] - 0.15, r[1] + 0.17], [r[0] - 0.15, r[1] - 0.17]], fc="none", ec="tab:blue", hatch="///", lw=1.2, zorder=5))
        rx_off = {1: (0.25, 0.05), 2: (0.25, -0.45), 3: (0.25, 0.35), 4: (0.25, -0.05)}[k]
        ax.text(r[0] + rx_off[0], r[1] + rx_off[1], f"WiFi Rx-{k} (approx.)\nCSI stored under node_{S.WIFI_RX_HOST_NODE[k]}", fontsize=7, color="tab:blue")
        ax.plot([tx[0], r[0]], [tx[1], r[1]], color="tab:blue", lw=0.6, ls="--", alpha=0.5, zorder=2)
    ax.add_patch(Polygon(obstacle_poly(), fc="gold", ec="k", lw=0.8, zorder=5))
    ax.text(S.OBSTACLE_APPROX["center"][0] - 1.3, S.OBSTACLE_APPROX["center"][1] - 0.15, "obstacle (3 cm board,\nNLOS to Rx-4; approx.)", fontsize=7)
    # subject
    ax.plot(0, 0, marker="*", ms=16, color="k", zorder=7)
    ax.text(0.15, -0.3, "subject spot = MoCap origin", fontsize=8)
    ax.set_xlim(-3.6, 4.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal")
    ax.set_xlabel("X [m]  (MoCap frame; +X = direction Node 1 faces from)"); ax.set_ylabel("Y [m]")
    ax.grid(alpha=0.3)
    ax.set_title("OctoNet Scene 1 - top-down layout (MoCap world frame, Z up)")
    handles = [Line2D([], [], marker="o", color="k", ls="", ms=9, label="Node 1-5: MEASURED (OptiTrack rigid bodies 'Node1'..'Node5', mean over the take)"),
               Line2D([], [], marker="^", color="tab:blue", ls="", ms=9, mfc="none", label="WiFi Tx/Rx, obstacle: SCHEMATIC from paper Fig. 2(a) (+-0.5 m)"),
               Line2D([], [], color="tab:green", lw=6, alpha=0.2, label="RGB-D FoV wedge: nominal D455 HFOV 87 deg, camera assumed to look at the origin"),
               Line2D([], [], color="tab:red", lw=1.4, ls=":", label="Vayyar SFCW R_max 3.71 m (node 1 only)"),
               Line2D([], [], marker="*", color="k", ls="", ms=12, label="subject standing spot (origin)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.95)
    fig.text(0.01, 0.005, "Node positions from mocap_csv_final/mocap_S1/chengxiao/'Take 2024-05-18 01.43.58 PM_subject_2.csv' (identical across Scene-1 takes; Scene 2/3 CSVs carry no node rigid bodies).\n"
             "Figure->MoCap axis mapping for the schematic items: fig. up = +X, fig. right = -Y.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "nodes_topdown_2d.png"), dpi=170); plt.close(fig)


def fig_3d():
    sk, skf = load_skeleton_frame()
    fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid([-HALF, HALF], [-HALF, HALF]); ax.plot_surface(xx, yy, np.zeros_like(xx), color="#e8f3e2", alpha=0.5, zorder=0)
    sq = np.array([[-HALF, -HALF], [HALF, -HALF], [HALF, HALF], [-HALF, HALF], [-HALF, -HALF]])
    ax.plot(sq[:, 0], sq[:, 1], 0, "k--", lw=0.8)
    for n, p in S.NODE_POS_S1.items():
        ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color=NODE_COL[n], lw=1.2, ls=":")
        ax.scatter(*p, s=140, color=NODE_COL[n], edgecolor="k", depthshade=False)
        ax.text(p[0], p[1], p[2] + 0.12, f"Node {n}  z={p[2]:.2f} m", fontsize=8)
        if n in CAM_NODES:   # boresight to the origin at node height
            ax.plot([p[0], 0], [p[1], 0], [p[2], p[2]], color=NODE_COL[n], lw=0.8, alpha=0.6)
    for k, r in S.WIFI_RX_APPROX.items():
        ax.scatter(*r, marker="^", s=70, color="tab:blue", alpha=0.8); ax.text(r[0], r[1], r[2] + 0.1, f"Rx-{k}~", fontsize=7, color="tab:blue")
    tx = S.WIFI_TX_APPROX; ax.scatter(*tx, marker="^", s=90, color="tab:red", alpha=0.8); ax.text(tx[0], tx[1], tx[2] + 0.1, "WiFi Tx~", fontsize=7, color="tab:red")
    ob = obstacle_poly(); ax.plot(np.r_[ob[:, 0], ob[0, 0]], np.r_[ob[:, 1], ob[0, 1]], S.OBSTACLE_APPROX["center"][2], color="gold", lw=3)
    ax.scatter(sk[:, 0], sk[:, 1], sk[:, 2], s=14, color="k", depthshade=False)
    for a, b in BONES:
        ax.plot([sk[a, 0], sk[b, 0]], [sk[a, 1], sk[b, 1]], [sk[a, 2], sk[b, 2]], color="k", lw=1.2)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_zlim(0, 2.2)
    ax.set_box_aspect((7, 7, 2.2))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m] (floor = 0)")
    ax.view_init(elev=28, azim=-135)
    ax.set_title("OctoNet Scene 1 - 3D layout: nodes at measured heights, MoCap skeleton at the origin for scale")
    fig.text(0.01, 0.01, f"Skeleton: tallest-neck frame of mocap_pose/{skf} (20 joints, mm->m, Z up; joint order inferred - bones indicative).\n"
             "'~' = schematic (paper Fig. 2a) position, height guessed 0.9 m. Nodes = measured OptiTrack rigid bodies.", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nodes_3d.png"), dpi=170); plt.close(fig)


COLMAP = {"vayyar(SFCW)": "vayyar", "uwb": "uwb", "wifi": "wifi", "depthCamera": "depthCamera", "mmWave(TI)": "mmWave",
          "IRA": "IRA", "ToF": "ToF", "acoustic": "acoustic", "seekThermal": "seekThermal", "polar(HR)": "polar"}


def fig_matrix():
    df = IO.load_index(); N = len(df)
    nodes = [1, 2, 3, 4, 5]; cols = S.SENSOR_COLUMNS
    avail = np.zeros((5, len(cols)), bool); cnt = np.zeros((5, len(cols)), int)
    for i, n in enumerate(nodes):
        for j, c in enumerate(cols):
            avail[i, j] = c in S.NODE_SENSORS[n]
            col = f"node_{n}_{COLMAP[c]}_data_path"
            if col in df.columns:
                cnt[i, j] = int(df[col].notna().sum())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(np.where(avail, 1.0, 0.0), cmap="Greens", vmin=0, vmax=1.6, aspect="auto")
    for i in range(5):
        for j in range(len(cols)):
            if avail[i, j]:
                ax.text(j, i, f"yes\n{cnt[i, j]}/{N}", ha="center", va="center", fontsize=8.5, color="k")
            else:
                ax.text(j, i, "-" if cnt[i, j] == 0 else f"dir missing\n{cnt[i, j]}/{N}", ha="center", va="center", fontsize=8.5, color="0.5")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels([f"node_{n}" for n in nodes])
    ax.set_title(f"OctoNet: sensor availability per node\ngreen = node_<n>/<sensor>/ exists on disk; number = cut_manual.csv recordings with a path, of {N}", fontsize=11)
    for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.01, 0.01, "Notes: Vayyar SFCW + UWB exist only on node_1; RGB-D (depthCamera) on nodes 1/3/5; WiFi receivers on nodes 1-4; ToF only node_4; every node has a TI mmWave radar and an IRA.\n"
             "cut_manual.csv also has 'uwb_data_path' (976) = node_1_uwb, 'mocap_data_path' (976), 'imu_data_path' (948).", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "node_sensor_matrix.png"), dpi=170); plt.close(fig)
    return cnt, N


def write_captions(cnt, N):
    txt = f"""nodes_topdown_2d.png
Top-down plan of OctoNet Scene 1 in the MoCap (OptiTrack) world frame (metres, Z up, origin = the subject's standing spot). Node 1-5 positions are MEASURED: they are the 'Node1'..'Node5' rigid bodies (3 reflective markers each, paper Sec. 3.2) exported in every Scene-1 OptiTrack CSV, averaged over the take (std 0 mm; identical within 3 mm across Scene-1 takes). Scene 2 and Scene 3 CSVs contain NO node rigid bodies, so only Scene 1 is drawn. Node 1 (the one that 'faces participants directly') is 2.92 m from the origin along +X at 0.90 m height; the other nodes sit at 0.79-0.93 m. The 4.1 m x 4.1 m sensing square (paper Fig. 2a) is ASSUMED centred on the origin. WiFi Tx, Rx-1..4 and the 3 cm wooden obstacle are NOT tracked by the MoCap and are drawn SCHEMATICALLY from paper Fig. 2(a) (hatched triangles; +-0.5 m), using the axis mapping figure-up = +X, figure-right = -Y; the CSI of receiver k is stored under node_k/wifi. RGB-D field-of-view wedges for nodes 1/3/5 use the NOMINAL RealSense D455 depth HFOV of 87 deg (intrinsics are not shipped) and assume each camera looks horizontally at the origin (no calibration exists yet). The dotted arc is the Vayyar SFCW unambiguous range R_max = 3.71 m from node 1; the UWB X4M200 (node 1) reaches 9.9 m, beyond the plot.

nodes_3d.png
Same layout in 3D with the floor plane at Z = 0 and the node mounting heights (dotted drop lines; values from the rigid bodies). Thin lines from nodes 1/3/5 to the origin are the assumed horizontal camera boresights. A 20-joint MoCap skeleton (mocap_pose/*_skeleton.npy, mm converted to m, Z up, floor-referenced) is drawn at the origin for scale, using the frame with the highest neck; the joint order (OptiTrack 21-bone skeleton without 'Head') was inferred from joint heights, so the bones are indicative only. '~' marks schematic (paper Fig. 2a) items whose height (0.9 m) is a guess.

node_sensor_matrix.png
Rows = node_1..5, columns = sensors. Green = the directory node_<n>/<sensor>/ exists in the dataset on disk (not the paper's table). The number in each cell is how many of the {N} recordings in cut_manual.csv carry a path for that node/sensor (e.g. node_1 vayyar {cnt[0,0]}, node_1 uwb {cnt[0,1]}, node_1 wifi {cnt[0,2]}, node_3 depthCamera {cnt[2,3]}, node_5 depthCamera {cnt[4,3]}). Vayyar SFCW and UWB exist only on node_1; RGB-D on nodes 1/3/5; WiFi receivers on nodes 1-4 (node_5 has none); ToF only on node_4; TI mmWave (FMCW point cloud) and IRA on all five. MoCap ({N} rows) and IMU are not node-bound.
"""
    open(os.path.join(OUT, "captions.txt"), "w").write(txt)


if __name__ == "__main__":
    fig_topdown(); print("wrote nodes_topdown_2d.png")
    fig_3d(); print("wrote nodes_3d.png")
    cnt, N = fig_matrix(); print("wrote node_sensor_matrix.png")
    write_captions(cnt, N); print("wrote captions.txt ->", OUT)
