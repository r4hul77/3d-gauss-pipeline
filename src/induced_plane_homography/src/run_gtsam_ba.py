#!/usr/bin/env python3
import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cv2
from vismatch import get_matcher
# --- Plotting (optional)
import tqdm
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- Your camera plotting helper (you said it's importable)
from rip_colmap_vismatch import plotCamera
# --- Your matcher factory (vismatch / romav2 wrapper)
# Expected interface: matcher = get_matcher(name, device="cuda"); matches = matcher(imgPIL, templatePIL_or_other)
# For BA we need matcher(img_i, img_j) -> dict with inlier_kpts0, inlier_kpts1 as Nx2 numpy arrays

# --- GTSAM
try:
    import gtsam
    from gtsam import symbol
except Exception as e:
    raise RuntimeError("gtsam import failed. Install python gtsam or fix env.") from e


# -------------------------
# Loading helpers
# -------------------------
def load_rip_colmap_pickle(obj: Any):
    """
    Supports either:
      - dict: expects keys like 'K','d','rvec','tvec' (or plural variants)
      - tuple/list from rip_colmap: (Hs, K, d, rvec, tvec, rep, Hm)
    Returns:
      K (3,3), d (Nx1 or (N,)), rvec_list, tvec_list
    """
    if isinstance(obj, dict):
        K = obj.get("K")
        d = obj.get("d")
        rvec = obj.get("rvec")
        tvec = obj.get("tvec")
        if K is None or rvec is None or tvec is None:
            raise KeyError("rip_colmap pickle dict must include at least K, rvec/tvec.")
        return np.asarray(K), (None if d is None else np.asarray(d)), rvec, tvec

    if isinstance(obj, (tuple, list)):
        # rip_colmap return order: Hs, K, d, rvec, tvec, rep, Hm
        if len(obj) < 5:
            raise ValueError("rip_colmap tuple/list too short.")
        K = np.asarray(obj[1])
        d = np.asarray(obj[2]) if obj[2] is not None else None
        rvec = obj[3]
        tvec = obj[4]
        return K, d, rvec, tvec

    raise TypeError(f"Unsupported rip_colmap pickle type: {type(obj)}")


def poses_from_rvec_tvec(rvec_list, tvec_list):
    """
    Convert OpenCV world->cam rvec/tvec into cam->world Pose3 init:
      R_c2w = inv(R_w2c)
      t_c2w = -R_c2w * t_w2c
    Returns:
      R_c2w: (N,3,3), t_c2w: (N,3)
    """
    N = len(rvec_list)
    R_c2w = np.zeros((N, 3, 3), dtype=np.float64)
    t_c2w = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        R_w2c, _ = cv2.Rodrigues(np.asarray(rvec_list[i]).reshape(3))
        t_w2c = np.asarray(tvec_list[i]).reshape(3, 1)

        R = np.linalg.inv(R_w2c)
        t = (-R @ t_w2c).reshape(3)

        R_c2w[i] = R
        t_c2w[i] = t

    return R_c2w, t_c2w


def load_edges_pickle(obj: Any):
    """
    Expects build_pose_graph.py output format:
      dict with keys: 'edges', 'R_c2w', 't_c2w' (poses optional here)
    """
    if not isinstance(obj, dict):
        raise TypeError("edges pickle must be a dict output by build_pose_graph.py")
    edges = obj.get("edges", None)
    if edges is None:
        raise KeyError("edges pickle dict must contain 'edges'")
    return edges, obj.get("R_c2w", None), obj.get("t_c2w", None)


# -------------------------
# Plot helpers
# -------------------------
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([2, 2, 2])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def plot_edges_3d(ax, t_c2w, edges, *, alpha=0.35, linewidth=0.8, by_stage=True):
    C = np.asarray(t_c2w, dtype=np.float64).reshape(-1, 3)
    stage_style = {
        "temporal": dict(linestyle="-",  alpha=alpha, linewidth=linewidth),
        "look":     dict(linestyle="--", alpha=alpha, linewidth=linewidth),
        "far":      dict(linestyle=":",  alpha=alpha, linewidth=linewidth),
    }
    for e in edges:
        i, j = int(e["i"]), int(e["j"])
        stg = e.get("stage", "temporal")
        p, q = C[i], C[j]
        if by_stage:
            kw = stage_style.get(stg, dict(alpha=alpha, linewidth=linewidth))
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **kw)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], alpha=alpha, linewidth=linewidth)


# -------------------------
# BA core
# -------------------------
def make_calibration_from_K(K: np.ndarray):
    """
    Create a GTSAM calibration model from OpenCV K.
    Uses Cal3_S2: fx, fy, s, cx, cy.
    """
    K = np.asarray(K, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    s  = float(K[0, 1])  # usually 0
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return gtsam.Cal3_S2(fx, fy, s, cx, cy)


def pose3_from_Rt(R_c2w: np.ndarray, t_c2w: np.ndarray) -> gtsam.Pose3:
    R = gtsam.Rot3(R_c2w)
    t = gtsam.Point3(float(t_c2w[0]), float(t_c2w[1]), float(t_c2w[2]))
    return gtsam.Pose3(R, t)


def triangulate_point(cameras: Dict[int, gtsam.PinholeCameraCal3_S2],
                      i: int, j: int,
                      zi: np.ndarray, zj: np.ndarray) -> Optional[gtsam.Point3]:
    """
    Triangulate from two views using current camera estimates.
    zi/zj are 2D pixel coords (u,v).
    """
    try:
        # GTSAM expects Point2(x,y)
        p_i = gtsam.Point2(float(zi[0]), float(zi[1]))
        p_j = gtsam.Point2(float(zj[0]), float(zj[1]))
        # Use built-in triangulation
        X = gtsam.triangulatePoint3(cameras[i], p_i, cameras[j], p_j, rankTol=1e-9, optimize=True)
        if np.any(np.isnan([X.x(), X.y(), X.z()])):
            return None
        return X
    except Exception as e:
        print(f"[triangulate_point] FAILED edge ({i},{j}) zi={zi} zj={zj}")
        print(f"  Exception: {type(e).__name__}: {e}")
        return None


def run_bundle_adjustment(
    edges: List[Dict[str, Any]],
    img_paths: List[str],
    K: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    matcher_name: str = "vismatch",
    device: str = "cuda",
    max_edges: int = -1,
    max_points_per_edge: int = 200,
    meas_sigma_px: float = 1.0,
    prior_sigma_rot_deg: float = 3.0,
    prior_sigma_trans: float = 0.05,
    add_pose_priors: bool = False,
):
    """
    Builds a BA graph:
      - Pose3 for each camera
      - For each edge (i,j), match images, triangulate points, add projection factors for i and j

    NOTE: Points are *not tracked across multiple edges* (duplicates are fine for debugging).
    """
    if get_matcher is None:
        raise RuntimeError("get_matcher import failed. Fix import at top (matchers.get_matcher).")

    # --- matcher
    matcher = get_matcher(matcher_name, device=device)

    # --- calibration
    calib = make_calibration_from_K(K)

    N = len(img_paths)

    # --- Graph + init
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Pose noise models
    rot_sig = np.deg2rad(prior_sigma_rot_deg)
    prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([rot_sig, rot_sig, rot_sig, prior_sigma_trans, prior_sigma_trans, prior_sigma_trans], dtype=np.float64)
    )
    meas_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(meas_sigma_px))

    # Initialize poses
    for i in range(N):
        Xi = symbol('x', i)
        T = pose3_from_Rt(R_init[i], t_init[i])
        initial.insert(Xi, T)

    # Prior on first pose (always)
    graph.add(gtsam.PriorFactorPose3(symbol('x', 0), initial.atPose3(symbol('x', 0)), prior_pose_noise))

    # Optional: priors on all poses (useful if your init is already global-good)
    if add_pose_priors:
        for i in range(1, N):
            graph.add(gtsam.PriorFactorPose3(symbol('x', i), initial.atPose3(symbol('x', i)), prior_pose_noise))

    # Build PinholeCamera objects for triangulation using init poses
    cameras_init = {}
    for i in range(N):
        cameras_init[i] = gtsam.PinholeCameraCal3_S2(initial.atPose3(symbol('x', i)), calib)

    # Iterate edges
    pt_counter = 0
    used_edges = edges if max_edges <= 0 else edges[:max_edges]

    # Lazy image loader
    from PIL import Image
    print(f"[BA] Running BA for {len(used_edges)} edges")
    for eidx, e in tqdm.tqdm(enumerate(used_edges)):
        i, j = int(e["i"]), int(e["j"])
        if i < 0 or j < 0 or i >= N or j >= N:
            continue

        # Load images
        img_i = Image.open(img_paths[i]).convert("RGB")
        img_j = Image.open(img_paths[j]).convert("RGB")

        # Run vismatch / matcher
        m = matcher(img_i, img_j)
        k0 = np.asarray(m["inlier_kpts0"], dtype=np.float32)  # (M,2)
        k1 = np.asarray(m["inlier_kpts1"], dtype=np.float32)  # (M,2)
        print(f"[BA] Edge {eidx} has {k0.shape[0]} inliers")
        if k0.shape[0] == 0:
            continue

        # Subsample to keep graph size sane
        M = k0.shape[0]
        if M > max_points_per_edge:
            idx = np.random.choice(M, size=max_points_per_edge, replace=False)
            k0 = k0[idx]
            k1 = k1[idx]

        Xi = symbol('x', i)
        Xj = symbol('x', j)

        for u0, u1 in zip(k0, k1):
            # Triangulate initial point from the two views (using init cameras)
            pose_vector = gtsam.Pose3Vector()
            pose_i = initial.atPose3(Xi)
            pose_j = initial.atPose3(Xj)
            pose_vector.append(pose_i)
            pose_vector.append(pose_j)

            meas_vector = gtsam.Point2Vector()
            m0 = gtsam.Point2(np.float64(u0[0]), np.float64(u0[1]))
            m1 = gtsam.Point2(np.float64(u1[0]), np.float64(u1[1]))
            meas_vector.append(m0)
            meas_vector.append(m1)
            try:
                X = gtsam.triangulatePoint3(
                        poses=pose_vector, 
                        sharedCal=calib, 
                        measurements=meas_vector, 
                        rank_tol=1e-9, 
                        optimize=True
                    )
            except Exception as e:
                print(f"[BA] Triangulation failed for edge {eidx} point {pt_counter}")
                print(f"  Exception: {type(e).__name__}: {e}")
                continue

            Pi = symbol('p', pt_counter)
            pt_counter += 1
            initial.insert(Pi, X)

            # Add projection factors for both observations
            z_i = gtsam.Point2(float(u0[0]), float(u0[1]))
            z_j = gtsam.Point2(float(u1[0]), float(u1[1]))

            graph.add(gtsam.GenericProjectionFactorCal3_S2(z_i, meas_noise, Xi, Pi, calib))
            graph.add(gtsam.GenericProjectionFactorCal3_S2(z_j, meas_noise, Xj, Pi, calib))

    print(f"[BA] Cameras: {N}, Edges used: {len(used_edges)}, Points created: {pt_counter}")

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # Extract results
    R_opt = np.zeros_like(R_init)
    t_opt = np.zeros_like(t_init)
    for i in range(N):
        T = result.atPose3(symbol('x', i))
        R_opt[i] = T.rotation().matrix()
        t_opt[i] = np.array([T.x(), T.y(), T.z()], dtype=np.float64)

    pts = []
    for k in range(pt_counter):
        Pk = symbol('p', k)
        if result.exists(Pk):
            X = result.atPoint3(Pk)
            pts.append([X])
    pts = np.asarray(pts, dtype=np.float64) if len(pts) else np.zeros((0, 3), dtype=np.float64)

    return R_opt, t_opt, pts, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rip_pickle", required=True, help="Pickle from rip_colmap(...) (K,rvec,tvec)")
    ap.add_argument("--edges_pickle", required=True, help="Pickle from build_pose_graph.py (edges)")
    ap.add_argument("--output_pickle", required=True, help="Output pickle with optimized poses + points")
    ap.add_argument("--img_glob_pickle_key", default="img_paths",
                    help="If rip_pickle is a dict, optional key for image paths list (default: img_paths)")
    ap.add_argument("--img_paths_txt", default="",
                    help="Optional: path to a text file listing image paths (one per line), if not in rip_pickle")
    ap.add_argument("--matcher", default="vismatch", help="Matcher name passed to get_matcher")
    ap.add_argument("--device", default="cuda", help="Device for matcher")
    ap.add_argument("--max_edges", type=int, default=-1, help="Limit number of edges matched (debug)")
    ap.add_argument("--max_points_per_edge", type=int, default=200, help="Cap points per edge")
    ap.add_argument("--meas_sigma_px", type=float, default=1.0, help="Pixel measurement noise sigma")
    ap.add_argument("--prior_sigma_rot_deg", type=float, default=3.0, help="Pose prior rotation sigma (deg)")
    ap.add_argument("--prior_sigma_trans", type=float, default=0.05, help="Pose prior translation sigma (world units)")
    ap.add_argument("--add_pose_priors", action="store_true", help="Add priors on all poses (not just pose0)")

    # Plot
    ap.add_argument("--plot", action="store_true", help="Plot cameras + edges + points after BA")
    ap.add_argument("--plot_every", type=int, default=2, help="Plot every kth camera")
    ap.add_argument("--cam_scale", type=float, default=0.05, help="Camera glyph scale for plotCamera")
    ap.add_argument("--edge_alpha", type=float, default=0.35, help="Edge alpha")
    ap.add_argument("--edge_linewidth", type=float, default=0.8, help="Edge linewidth")
    ap.add_argument("--by_stage", action="store_true", help="Style edges by stage")
    ap.add_argument("--point_stride", type=int, default=1, help="Plot every kth point (debug)")
    args = ap.parse_args()

    # Load pickles
    with open(args.rip_pickle, "rb") as f:
        rip_data = pickle.load(f)
    with open(args.edges_pickle, "rb") as f:
        edges_data = pickle.load(f)

    K, d, rvec_list, tvec_list = load_rip_colmap_pickle(rip_data)
    edges, R0_from_edges, t0_from_edges = load_edges_pickle(edges_data)

    # Image paths discovery:
    img_paths = None

    # 1) If rip_pickle is dict and has img_paths stored
    if isinstance(rip_data, dict) and args.img_glob_pickle_key in rip_data:
        img_paths = list(rip_data[args.img_glob_pickle_key])

    # 2) else try a provided txt list
    if img_paths is None and args.img_paths_txt.strip():
        with open(args.img_paths_txt, "r") as f:
            img_paths = [ln.strip() for ln in f.readlines() if ln.strip()]

    if img_paths is None:
        raise RuntimeError(
            "Could not find image paths. Either store them in rip_pickle dict under key "
            f"'{args.img_glob_pickle_key}' OR pass --img_paths_txt."
        )

    # Initial poses: prefer build_pose_graph poses if present, else derive from rvec/tvec
    if R0_from_edges is not None and t0_from_edges is not None:
        R_init = np.asarray(R0_from_edges, dtype=np.float64)
        t_init = np.asarray(t0_from_edges, dtype=np.float64).reshape(-1, 3)
    else:
        R_init, t_init = poses_from_rvec_tvec(rvec_list, tvec_list)

    # Run BA
    R_opt, t_opt, pts, gtsam_result = run_bundle_adjustment(
        edges=edges,
        img_paths=img_paths,
        K=K,
        R_init=R_init,
        t_init=t_init,
        matcher_name=args.matcher,
        device=args.device,
        max_edges=args.max_edges,
        max_points_per_edge=args.max_points_per_edge,
        meas_sigma_px=args.meas_sigma_px,
        prior_sigma_rot_deg=args.prior_sigma_rot_deg,
        prior_sigma_trans=args.prior_sigma_trans,
        add_pose_priors=args.add_pose_priors,
    )

    out = {
        "K": K,
        "R_opt": R_opt,
        "t_opt": t_opt,
        "points": pts,
        "edges": edges,
        "params": vars(args),
    }

    with open(args.output_pickle, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[BA] Wrote: {args.output_pickle}")
    print(f"[BA] Points: {pts.shape[0]}")

    # Plot
    if args.plot:
        if plt is None:
            raise RuntimeError("matplotlib not available.")
        if plotCamera is None:
            raise RuntimeError("plotCamera import failed; fix it at the top of this script.")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("BA result: cameras + edges + points")

        # Cameras (subsample)
        for i in range(0, len(R_opt), max(1, int(args.plot_every))):
            plotCamera(ax, R_opt[i], t_opt[i], scale=args.cam_scale, is_w2c=False)

        # Edges
        plot_edges_3d(ax, t_opt, edges, alpha=args.edge_alpha, linewidth=args.edge_linewidth, by_stage=args.by_stage)

        # Points
        if pts.shape[0] > 0:
            stride = max(1, int(args.point_stride))
            P = pts[::stride]
            ax.scatter(P[:500,0, 0], P[:500,0, 1], P[:500,0, 2], s=2.0, alpha=0.6)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
