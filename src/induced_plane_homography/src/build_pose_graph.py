#!/usr/bin/env python3
import argparse
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


from rip_colmap_vismatch import plotCamera


def edge_key(i: int, j: int) -> Tuple[int, int]:
    """Canonical undirected key: (min,max). Prevents i->j and j->i duplicates."""
    if i == j:
        raise ValueError("Self-edge requested.")
    return (i, j) if i < j else (j, i)


@dataclass
class EdgeData:
    i: int
    j: int
    stage: str               # "temporal" | "look" | "far"
    score: float             # dot score (or 0 for temporal)
    reason: str = ""         # optional debug message


def load_rvec_tvec(obj):
    """
    Supports either:
      - dict with keys: 'rvec', 'tvec' (or 'rvecs','tvecs')
      - tuple/list returned from rip_colmap: (Hs, K, d, rvec, tvec, rep, Hm)
    """
    if isinstance(obj, dict):
        rvec = obj.get("rvec", None) or obj.get("rvecs", None)
        tvec = obj.get("tvec", None) or obj.get("tvecs", None)
        if rvec is None or tvec is None:
            raise KeyError("Pickle dict must contain rvec/tvec (or rvecs/tvecs).")
        return rvec, tvec

    if isinstance(obj, (tuple, list)):
        # Based on your return order: Hs, K, d, rvec, tvec, rep, Hm
        if len(obj) < 5:
            raise ValueError("Pickle tuple/list too short to contain rvec/tvec.")
        rvec = obj[3]
        tvec = obj[4]
        return rvec, tvec

    raise TypeError(f"Unsupported pickle type: {type(obj)}")


def poses_from_rvec_tvec(rvec_list, tvec_list):
    """
    Convert OpenCV rvec/tvec (world->cam) to cam->world:
      R_c2w = inv(R_w2c)
      t_c2w = -R_c2w * t_w2c

    Returns:
      R_c2w: (N,3,3)
      t_c2w: (N,3)
    """
    N = len(rvec_list)
    R_c2w = np.zeros((N, 3, 3), dtype=np.float64)
    t_c2w = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        R_w2c, _ = cv2.Rodrigues(np.asarray(rvec_list[i]).reshape(3))
        t_w2c = np.asarray(tvec_list[i]).reshape(3, 1)

        R = np.linalg.inv(R_w2c)              # cam->world
        t = (-R @ t_w2c).reshape(3)           # cam origin in world

        R_c2w[i] = R
        t_c2w[i] = t

    return R_c2w, t_c2w


def compute_forward_axis_world(R_c2w: np.ndarray, axis: str = "z") -> np.ndarray:
    """
    Compute camera forward axis in world coordinates using R_c2w.

    axis options: 'x', 'y', 'z', '-x', '-y', '-z'
    """
    axis_map = {
        "x":  np.array([1.0, 0.0, 0.0]),
        "y":  np.array([0.0, 1.0, 0.0]),
        "z":  np.array([0.0, 0.0, 1.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Choose from {list(axis_map.keys())}")

    v_cam = axis_map[axis].reshape(3, 1)  # (3,1)
    N = R_c2w.shape[0]
    fwd = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        f = (R_c2w[i] @ v_cam).reshape(3)
        n = np.linalg.norm(f) + 1e-12
        fwd[i] = f / n

    return fwd


def add_edge(
    edges: Dict[Tuple[int, int], EdgeData],
    neighbors: List[set],
    i: int,
    j: int,
    stage: str,
    score: float,
    reason: str = "",
):
    k = edge_key(i, j)
    if k in edges:
        return False
    edges[k] = EdgeData(i=k[0], j=k[1], stage=stage, score=float(score), reason=reason)
    neighbors[k[0]].add(k[1])
    neighbors[k[1]].add(k[0])
    return True


def build_edges(
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
    temporal_radius: int = 2,
    k_extra: int = 5,
    far_offsets: Optional[List[int]] = None,
    k_far: int = 5,
    forward_axis: str = "z",
    degree_cap: Optional[int] = None,
    distance_gate: Optional[Tuple[float, float]] = None,  # (dmin, dmax)
) -> Dict[Tuple[int, int], EdgeData]:
    """
    Builds undirected edges with no duplicates:
      Step A: temporal j in [i-2, i-1, i+1, i+2] (temporal_radius=2)
      Step B: per-node add up to k_extra edges maximizing dot(forward_i, forward_j) among non-neighbors
      Step C: "far" edges using far_offsets (default [5] => ±5), per-node up to k_far
    """
    t_c2w = np.asarray(t_c2w, dtype=np.float64).reshape(-1, 3)
    N = R_c2w.shape[0]
    neighbors = [set() for _ in range(N)]
    edges: Dict[Tuple[int, int], EdgeData] = {}

    fwd = compute_forward_axis_world(R_c2w, axis=forward_axis)

    def under_cap(a: int, b: int) -> bool:
        if degree_cap is None:
            return True
        return (len(neighbors[a]) < degree_cap) and (len(neighbors[b]) < degree_cap)

    def passes_dist(i: int, j: int) -> bool:
        if distance_gate is None:
            return True
        dmin, dmax = distance_gate
        d = float(np.linalg.norm(t_c2w[i] - t_c2w[j]))
        return (d >= dmin) and (d <= dmax)

    # Step A: Temporal neighbors
    for i in range(N):
        for dj in range(-temporal_radius, temporal_radius + 1):
            if dj == 0:
                continue
            j = i + dj
            if 0 <= j < N and under_cap(i, j):
                add_edge(edges, neighbors, i, j, stage="temporal", score=0.0, reason="local window")

    # Step B: Look-direction max dot among non-neighbor nodes
    for i in range(N):
        added = 0
        cand = []
        for j in range(N):
            if j == i:
                continue
            if j in neighbors[i]:
                continue
            if not under_cap(i, j):
                continue
            if not passes_dist(i, j):
                continue
            score = float(np.dot(fwd[i], fwd[j]))
            cand.append((score, j))

        cand.sort(reverse=True, key=lambda x: x[0])

        for score, j in cand:
            if added >= k_extra:
                break
            ok = add_edge(edges, neighbors, i, j, stage="look", score=score, reason="max dot(fwd)")
            if ok:
                added += 1

    # Step C: Far edges
    if far_offsets is None:
        far_offsets = [5]

    for i in range(N):
        added = 0
        far_js = set()
        for off in far_offsets:
            for j in (i - off, i + off):
                if 0 <= j < N:
                    far_js.add(j)

        cand = []
        for j in sorted(far_js):
            if j == i or j in neighbors[i]:
                continue
            if not under_cap(i, j):
                continue
            if not passes_dist(i, j):
                continue
            score = float(np.dot(fwd[i], fwd[j]))
            cand.append((score, j))

        cand.sort(reverse=True, key=lambda x: x[0])

        for score, j in cand:
            if added >= k_far:
                break
            ok = add_edge(edges, neighbors, i, j, stage="far", score=score, reason=f"far offsets {far_offsets}")
            if ok:
                added += 1

    return edges


# -------------------------
# Plotting helpers
# -------------------------
def set_axes_equal(ax):
    """Make 3D axes equal scale so trajectory isn't distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def plot_pose_graph_edges(ax, t_c2w, edges_list, *, alpha=0.35, linewidth=0.8, by_stage=False):
    """
    Draw undirected edges between camera centers.
    edges_list: list of dicts with keys i,j,(optional stage) OR list of tuples (i,j)
    """
    C = np.asarray(t_c2w, dtype=np.float64).reshape(-1, 3)

    stage_style = {
        "temporal": dict(linestyle="-",  alpha=alpha, linewidth=linewidth),
        "look":     dict(linestyle="--", alpha=alpha, linewidth=linewidth),
        "far":      dict(linestyle=":",  alpha=alpha, linewidth=linewidth),
    }

    for e in edges_list:
        if isinstance(e, dict):
            i, j = int(e["i"]), int(e["j"])
            stg = e.get("stage", "temporal")
        else:
            i, j = int(e[0]), int(e[1])
            stg = "temporal"

        p = C[i]
        q = C[j]

        if by_stage:
            kw = stage_style.get(stg, dict(alpha=alpha, linewidth=linewidth))
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **kw)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], alpha=alpha, linewidth=linewidth)


def plot_pose_graph_with_cameras(R_c2w, t_c2w, edges_list, *,
                                 cam_scale=0.05,
                                 every=1,
                                 edge_alpha=0.35,
                                 edge_linewidth=0.8,
                                 by_stage=True,
                                 title="Pose graph (cameras + edges)"):
    if plt is None:
        raise RuntimeError("matplotlib is not available, cannot plot. Install it or run without --plot.")
    if plotCamera is None:
        raise RuntimeError("plotCamera import failed. Fix the import at top of the script.")

    R = np.asarray(R_c2w, dtype=np.float64)
    t = np.asarray(t_c2w, dtype=np.float64).reshape(-1, 3)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Cameras (subsample for speed)
    for i in range(0, len(R), int(every)):
        plotCamera(ax, R[i], t[i], scale=cam_scale, is_w2c=False)

    # Edges
    plot_pose_graph_edges(
        ax, t, edges_list,
        alpha=edge_alpha,
        linewidth=edge_linewidth,
        by_stage=by_stage
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pickle", required=True, help="Pickle output from rip_colmap(...)")
    ap.add_argument("--output_pickle", required=True, help="Where to write edges + poses pickle")
    ap.add_argument("--temporal_radius", type=int, default=2, help="Temporal radius (2 => i±1,i±2)")
    ap.add_argument("--k_extra", type=int, default=5, help="Extra look-direction edges per node")
    ap.add_argument("--far_offsets", type=str, default="5", help="Comma list, e.g. '5,10'")
    ap.add_argument("--k_far", type=int, default=5, help="Far edges per node (from far_offsets)")
    ap.add_argument("--forward_axis", type=str, default="z", help="Camera forward axis in camera frame: x,y,z,-x,-y,-z")
    ap.add_argument("--degree_cap", type=int, default=-1, help="Max degree per node (<=0 disables)")
    ap.add_argument("--distance_gate", type=str, default="", help="Optional 'dmin,dmax' in world units")

    # Plot options
    ap.add_argument("--plot", action="store_true", help="Show debug plot of cameras + edges")
    ap.add_argument("--plot_every", type=int, default=2, help="Plot every kth camera (speed)")
    ap.add_argument("--cam_scale", type=float, default=0.05, help="Camera glyph scale for plotCamera")
    ap.add_argument("--edge_alpha", type=float, default=0.35, help="Edge line alpha")
    ap.add_argument("--edge_linewidth", type=float, default=0.8, help="Edge line width")
    ap.add_argument("--by_stage", action="store_true", help="Style edges by stage (temporal/look/far)")
    args = ap.parse_args()

    with open(args.input_pickle, "rb") as f:
        data = pickle.load(f)

    rvec, tvec = load_rvec_tvec(data)
    R_c2w, t_c2w = poses_from_rvec_tvec(rvec, tvec)

    far_offsets = [int(x) for x in args.far_offsets.split(",") if x.strip() != ""]
    degree_cap = None if args.degree_cap <= 0 else int(args.degree_cap)

    distance_gate = None
    if args.distance_gate.strip():
        parts = [float(x) for x in args.distance_gate.split(",")]
        if len(parts) != 2:
            raise ValueError("--distance_gate must be 'dmin,dmax'")
        distance_gate = (parts[0], parts[1])

    edges = build_edges(
        R_c2w=R_c2w,
        t_c2w=t_c2w,
        temporal_radius=args.temporal_radius,
        k_extra=args.k_extra,
        far_offsets=far_offsets,
        k_far=args.k_far,
        forward_axis=args.forward_axis,
        degree_cap=degree_cap,
        distance_gate=distance_gate,
    )

    edges_list = [asdict(ed) for ed in edges.values()]

    out = {
        "R_c2w": R_c2w,
        "t_c2w": t_c2w,
        "edges": edges_list,
        "params": {
            "temporal_radius": args.temporal_radius,
            "k_extra": args.k_extra,
            "far_offsets": far_offsets,
            "k_far": args.k_far,
            "forward_axis": args.forward_axis,
            "degree_cap": degree_cap,
            "distance_gate": distance_gate,
        },
    }

    with open(args.output_pickle, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # quick stats
    N = R_c2w.shape[0]
    deg = [0] * N
    for ed in edges.values():
        deg[ed.i] += 1
        deg[ed.j] += 1
    print(f"N poses: {N}")
    print(f"Edges: {len(edges)}")
    print(f"Degree min/mean/max: {min(deg)}/{(sum(deg)/len(deg)):.2f}/{max(deg)}")

    if args.plot:
        plot_pose_graph_with_cameras(
            R_c2w, t_c2w, edges_list,
            cam_scale=args.cam_scale,
            every=max(1, int(args.plot_every)),
            edge_alpha=args.edge_alpha,
            edge_linewidth=args.edge_linewidth,
            by_stage=args.by_stage,
            title="Pose graph (cameras + edges)"
        )


if __name__ == "__main__":
    main()
