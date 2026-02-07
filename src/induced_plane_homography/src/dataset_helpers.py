from pathlib import Path
import shutil
import numpy as np
import cv2
import pickle
import os
from scipy.spatial.transform import Rotation as Rot

# ---------- quaternion helper ----------
def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    r = Rot.from_matrix(R)
    quat = r.as_quat()  # Returns [x, y, z, w]
    # Reorder to [w, x, y, z] for COLMAP format
    return np.array([quat[3], quat[0], quat[1], quat[2]])


# ---------- backend undistortion ----------
def _undistort_to_pinhole_backend(
    img_paths: list[str],
    K: np.ndarray,
    d: np.ndarray,
    out_images_dir: Path,
    *,
    alpha: float = 0.0,     # 0=crop valid, 1=keep all (black borders)
    crop: bool = True,      # if alpha=0 and crop=True, crop to ROI and shift cx/cy accordingly
    interpolation: int = cv2.INTER_LINEAR,
):
    """
    Undistort images to pinhole model.
    
    Returns:
        undist_paths: List of undistorted image paths
        K_used: Updated camera matrix
        out_wh: Output image size (W, H)
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)

    im0 = cv2.imread(img_paths[0])
    if im0 is None:
        raise RuntimeError(f"Could not read first image: {img_paths[0]}")
    H0, W0 = im0.shape[:2]

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(d, dtype=np.float64).reshape(-1)

    K_new, roi = cv2.getOptimalNewCameraMatrix(K, d, (W0, H0), alpha, (W0, H0))
    map1, map2 = cv2.initUndistortRectifyMap(K, d, None, K_new, (W0, H0), cv2.CV_16SC2)

    undist_paths = []
    # default output size/intrinsics if not cropping
    out_wh = (W0, H0)
    K_used = K_new.copy()

    if crop and alpha == 0.0:
        x, y, w, h = roi
        out_wh = (w, h)
        K_used = K_new.copy()
        K_used[0, 2] -= x
        K_used[1, 2] -= y

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: Could not read {p}, skipping...")
            continue
        und = cv2.remap(img, map1, map2, interpolation=interpolation)

        if crop and alpha == 0.0:
            x, y, w, h = roi
            und = und[y:y+h, x:x+w].copy()

        out_path = out_images_dir / Path(p).name
        cv2.imwrite(str(out_path), und)
        undist_paths.append(str(out_path))

    return undist_paths, K_used, out_wh


# ---------- main exporter ----------
def export_colmap_text_model(
    out_dir: str,
    img_paths: list[str],
    K: np.ndarray,
    d: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    image_size_wh: tuple[int, int] | None = None,
    camera_model: str = "FULL_OPENCV",
    copy_images: bool = False,
):
    """
    Export camera calibration to COLMAP text format.
    
    Args:
        out_dir: Output directory
        img_paths: List of image paths
        K: Camera intrinsic matrix (3x3)
        d: Distortion coefficients
        rvecs: List of rotation vectors (one per image)
        tvecs: List of translation vectors (one per image)
        image_size_wh: Image size as (width, height)
        camera_model: Camera model (PINHOLE, SIMPLE_PINHOLE, OPENCV_FISHEYE, or FULL_OPENCV)
        copy_images: Whether to copy images instead of symlinking
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    sparse0_dir = out_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse0_dir.mkdir(parents=True, exist_ok=True)

    supported = {"PINHOLE", "SIMPLE_PINHOLE", "OPENCV_FISHEYE"}
    requested = camera_model.upper()

    # -------------------------
    # Normalize to supported format
    # -------------------------
    export_img_paths: list[str]
    export_model: str
    export_K: np.ndarray
    export_d: np.ndarray
    export_wh: tuple[int, int]

    if requested in supported:
        export_model = requested
        export_img_paths = [str(p) for p in img_paths]
        export_K = np.asarray(K, dtype=np.float64)
        export_d = np.asarray(d, dtype=np.float64)

        if image_size_wh is None:
            im0 = cv2.imread(str(img_paths[0]))
            if im0 is None:
                raise RuntimeError(f"Could not read first image: {img_paths[0]}")
            H0, W0 = im0.shape[:2]
            export_wh = (W0, H0)
        else:
            export_wh = image_size_wh

    else:
        # Save originals BEFORE undistorting
        print(f"Camera model {requested} not directly supported. Converting to PINHOLE...")
        og_dir = out_dir / "og_imgs"
        og_dir.mkdir(parents=True, exist_ok=True)

        og_paths: list[str] = []
        for p in img_paths:
            src = Path(p)
            dst = og_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            og_paths.append(str(dst))

        # Convert to supported format: PINHOLE
        und_dir = out_dir / "_undistorted_images"
        und_paths, K_new, (W, H) = _undistort_to_pinhole_backend(
            og_paths, K, d, und_dir,
            alpha=0.0,
            crop=True,
        )

        export_model = "PINHOLE"
        export_img_paths = [str(p) for p in und_paths]
        export_K = np.asarray(K_new, dtype=np.float64)
        export_d = np.zeros((1, 1), dtype=np.float64)  # not used for PINHOLE
        export_wh = (int(W), int(H))

    # -------------------------
    # Compute COLMAP camera params from normalized model
    # -------------------------
    W, H = export_wh
    fx, fy = float(export_K[0, 0]), float(export_K[1, 1])
    cx, cy = float(export_K[0, 2]), float(export_K[1, 2])

    if export_model == "SIMPLE_PINHOLE":
        model_name = "SIMPLE_PINHOLE"
        cam_params = [fx, cx, cy]
    elif export_model == "PINHOLE":
        model_name = "PINHOLE"
        cam_params = [fx, fy, cx, cy]
    elif export_model == "OPENCV_FISHEYE":
        dd = np.asarray(export_d, dtype=np.float64).reshape(-1)
        dd = np.pad(dd, (0, max(0, 4 - dd.size)), mode="constant")[:4]
        model_name = "OPENCV_FISHEYE"
        cam_params = [fx, fy, cx, cy, float(dd[0]), float(dd[1]), float(dd[2]), float(dd[3])]
    else:
        raise RuntimeError(f"Internal error: normalized to unsupported model: {export_model}")

    # -------------------------
    # Export once (single writer)
    # -------------------------
    _write_colmap_text_model(
        images_dir=images_dir,
        sparse0_dir=sparse0_dir,
        img_paths=export_img_paths,
        rvecs=rvecs,
        tvecs=tvecs,
        model_name=model_name,
        W=W,
        H=H,
        cam_params=cam_params,
        copy_images=copy_images,
    )


def _write_colmap_text_model(
    images_dir: Path,
    sparse0_dir: Path,
    img_paths: list[str],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    model_name: str,
    W: int,
    H: int,
    cam_params: list[float],
    copy_images: bool,
):
    """
    Single exporter: writes cameras.txt, images.txt, points3D.txt, and materializes images/.
    """
    if len(img_paths) != len(rvecs) or len(img_paths) != len(tvecs):
        raise ValueError(
            f"Length mismatch: img_paths={len(img_paths)}, rvecs={len(rvecs)}, tvecs={len(tvecs)}"
        )

    cam_id = 1

    # cameras.txt
    with (sparse0_dir / "cameras.txt").open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            f"{cam_id} {model_name} {W} {H} "
            + " ".join(f"{p:.12g}" for p in cam_params)
            + "\n"
        )

    # images.txt
    with (sparse0_dir / "images.txt").open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(img_paths)}\n")

        for i, (p, rvec, tvec) in enumerate(zip(img_paths, rvecs, tvecs), start=1):
            # Pose conversion: cv2.calibrateCamera returns world-to-camera (w2c)
            # COLMAP ALSO expects world-to-camera (w2c) - no conversion needed!
            R_w2c, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
            tvec_w2c = np.asarray(tvec).reshape(3)

            q = rotmat_to_qvec(R_w2c)  # (qw, qx, qy, qz) from w2c rotation
            t = tvec_w2c  # w2c translation

            name = Path(p).name
            dst = images_dir / name
            if not dst.exists():
                if copy_images:
                    shutil.copy2(p, dst)
                else:
                    try:
                        dst.symlink_to(Path(p).resolve())
                    except Exception:
                        shutil.copy2(p, dst)

            f.write(
                f"{i} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} "
                f"{cam_id} {name}\n\n"
            )

    # points3D.txt (empty)
    (sparse0_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n"
    )

    print(f"[OK] COLMAP model written to: {sparse0_dir}")
    print(f"[OK] images/: {images_dir}")


def save_calibration_pickle(Hs, K, d, rvec, tvec, rep, file_path, good_imgs):
    """
    Save calibration results to a pickle file.
    
    Args:
        Hs: List of homography matrices
        K: Camera intrinsic matrix
        d: Distortion coefficients
        rvec: List of rotation vectors
        tvec: List of translation vectors
        rep: Reprojection error
        file_path: Output pickle file path
        good_imgs: List of successfully calibrated image paths
    """
    data = {
        "Hs": Hs,
        "K": K,
        "d": d,
        "rvec": rvec,
        "tvec": tvec,
        "rep": rep,
        "good_imgs": good_imgs
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Calibration saved to {file_path}")


def load_calibration_pickle(file_path):
    """
    Load calibration results from a pickle file.
    
    Returns:
        dict with keys: Hs, K, d, rvec, tvec, rep, good_imgs
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data