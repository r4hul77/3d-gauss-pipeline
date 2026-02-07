import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


import numpy as np
import matplotlib.pyplot as plt

def plotPlane(ax, n, d, bounds=((-1, 1), (-1, 1), (-1, 1)), resolution=20, alpha=0.35):
    """
    Plot plane n^T x = d on a matplotlib 3D axis.

    Args:
      ax: a matplotlib 3D axis (e.g., fig.add_subplot(111, projection='3d'))
      n: (3,) plane normal (doesn't need to be unit)
      d: scalar plane offset
      bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax))
      resolution: grid resolution along the two free axes
      alpha: surface transparency
    """
    n = np.asarray(n, dtype=np.float64).reshape(3)
    d = float(d)

    # Normalize for numerical stability (not required, but helps)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError("Normal vector is near-zero.")
    n = n / nn
    d = d / nn

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    # Choose which variable to solve for based on largest normal component
    k = int(np.argmax(np.abs(n)))  # 0->x,1->y,2->z

    if k == 2:
        # Solve for z: n_x x + n_y y + n_z z = d
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(xs, ys)
        if abs(n[2]) < 1e-12:
            raise ValueError("n_z is too small to solve for z.")
        Z = (d - n[0]*X - n[1]*Y) / n[2]
        ax.plot_surface(X, Y, Z, alpha=alpha, rstride=1, cstride=1, linewidth=0)

    elif k == 1:
        # Solve for y
        xs = np.linspace(xmin, xmax, resolution)
        zs = np.linspace(zmin, zmax, resolution)
        X, Z = np.meshgrid(xs, zs)
        if abs(n[1]) < 1e-12:
            raise ValueError("n_y is too small to solve for y.")
        Y = (d - n[0]*X - n[2]*Z) / n[1]
        ax.plot_surface(X, Y, Z, alpha=alpha, rstride=1, cstride=1, linewidth=0)

    else:  # k == 0
        # Solve for x
        ys = np.linspace(ymin, ymax, resolution)
        zs = np.linspace(zmin, zmax, resolution)
        Y, Z = np.meshgrid(ys, zs)
        if abs(n[0]) < 1e-12:
            raise ValueError("n_x is too small to solve for x.")
        X = (d - n[1]*Y - n[2]*Z) / n[0]
        ax.plot_surface(X, Y, Z, alpha=alpha, rstride=1, cstride=1, linewidth=0)



def plotCamera(ax, R, t, *, color=None, scale=1, width=2, height=1.5, focal_length=3, label=None, is_w2c=False, legend=None):
    """Plot a camera in 3D

    Args:
        ax (Axis3D): target axis
        R: 3x3 c2w rotation matrix (interpreted as w2c if is_w2c is set)
        t: 3x1 c2w translation vector (interpreted as w2c if is_w2c is set)
        c: color string
        scale: scaling factor
        width: horizontal size of the camera screen
        height: vertical size of the camera screen
        focal_length: height of the camera cone
        label: text at the camera position
        legend: text shown in the legend
        is_w2c: set True if R, t are w2c
    """

    t = t.reshape((3, 1))

    if is_w2c:
        R = R.T
        t = -R@t

    w = width
    h = height
    f = focal_length

    # focus, br, tr, tl, bl, triangle
    ps_c = np.array(([0,0,0], [w/2,h/2,f], [w/2,-h/2,f], [-w/2,-h/2,f], [-w/2,h/2,f], [0,-h/2-h/4,f]))
    ps_w = (scale * R @ ps_c.T + t).T

    L01 = np.array([ps_w[0], ps_w[1]])
    L02 = np.array([ps_w[0], ps_w[2]])
    L03 = np.array([ps_w[0], ps_w[3]])
    L04 = np.array([ps_w[0], ps_w[4]])
    L1234 = np.array([ps_w[1], ps_w[2], ps_w[3], ps_w[4], ps_w[1]])
    L253 = np.array([ps_w[2], ps_w[5], ps_w[3]])

    p = ax.plot(L01[:,0], L01[:,1], L01[:,2], "-", color=color, label=legend)
    if color is None:
        color = p[-1].get_color()
    ax.plot(L02[:,0], L02[:,1], L02[:,2], "-", color=color)
    ax.plot(L03[:,0], L03[:,1], L03[:,2], "-", color=color)
    ax.plot(L04[:,0], L04[:,1], L04[:,2], "-", color=color)
    ax.plot(L1234[:,0], L1234[:,1], L1234[:,2], "-", color=color)
    ax.plot(L253[:,0], L253[:,1], L253[:,2], "-", color=color)

    if label is not None:
        ax.text(ps_w[0][0], ps_w[0][1], ps_w[0][2], label)

# -----------------------
# Helpers
# -----------------------
def template_keypoints2objpoints(keypoints_template_xy: np.ndarray) -> np.ndarray:
    """
    Convert template 2D keypoints (x,y) into pseudo object points (x,y,0).
    Planar assumption.
    """
    keypoints_template_xy = np.asarray(keypoints_template_xy, dtype=np.float32).reshape(-1, 2)
    z = np.zeros((keypoints_template_xy.shape[0], 1), dtype=np.float32)
    obj = np.hstack([keypoints_template_xy, z])   # (N,3)
    return obj.reshape(-1, 1, 3)

def normalize_template_xy(kptsB_xy: np.ndarray, template_shape_hw, template_size: float):
    """
    Match your RoMaV2 normalization:
      kptsB = 0.5*template_size*kptsB
      kptsB = kptsB/mean([H_B, W_B]) - 0.5*template_size

    Here kptsB_xy are in pixel coords of the template.
    """
    H_B, W_B = template_shape_hw
    k = kptsB_xy.astype(np.float32)

    # scale by 0.5*template_size
    k = template_size * k

    # normalize by mean dim and shift
    k = k / np.mean([H_B, W_B]) - 0.5 * template_size
    return k

# -----------------------
# SIFT matching
# -----------------------
def sift_match_keypoints(img_bgr, template_bgr, *, ratio=0.75, max_kpts=8000):
    """
    Returns:
      kptsA_xy: (N,2) float32 in img pixel coords
      kptsB_xy: (N,2) float32 in template pixel coords
    """
    # grayscale
    img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tpl_g = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # SIFT (requires opencv-contrib)
    sift = cv2.SIFT_create(nfeatures=max_kpts)

    kpA, desA = sift.detectAndCompute(img_g, None)
    kpB, desB = sift.detectAndCompute(tpl_g, None)

    if desA is None or desB is None or len(kpA) < 8 or len(kpB) < 8:
        return None, None

    # FLANN for SIFT descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # knn match + ratio test
    knn = flann.knnMatch(desA, desB, k=2)

    good = []
    for matches in knn:
        if len(matches) == 2:  # Fix: ensure we have 2 matches
            m, n = matches
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 8:
        return None, None

    # extract matched coords
    kptsA_xy = np.array([kpA[m.queryIdx].pt for m in good], dtype=np.float32)
    kptsB_xy = np.array([kpB[m.trainIdx].pt for m in good], dtype=np.float32)
    return kptsA_xy, kptsB_xy

def ransac_homography(kptsA_xy, kptsB_xy, *, reproj_thresh=3.0, confidence=0.999, max_iters=5000):
    """
    Estimate H mapping template->image or image->template depending on order.
    We want H such that: kptsB(template) -> kptsA(image)
    """
    H, inliers = cv2.findHomography(
        kptsB_xy,  # src (template)
        kptsA_xy,  # dst (image)
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
        maxIters=max_iters,
        confidence=confidence,
    )
    if H is None or inliers is None:
        return None, None
    inliers = inliers.ravel().astype(bool)
    return H, inliers

# -----------------------
# Multiprocessing worker
# -----------------------
def process_single_image(img_path, template_path, template_shape_hw, template_size, 
                         ratio, reproj_thresh, min_inliers):
    """
    Worker function for multiprocessing. Processes a single image.
    
    Returns:
        dict with keys: 'success', 'img_path', 'H', 'img_inl', 'obj_inl', 'img_shape', 'num_inliers'
    """
    result = {
        'success': False,
        'img_path': img_path,
        'H': None,
        'img_inl': None,
        'obj_inl': None,
        'img_shape': None,
        'num_inliers': 0
    }
    
    try:
        # Load images
        img = cv2.imread(img_path)
        template = cv2.imread(template_path)
        
        if img is None or template is None:
            return result
        
        result['img_shape'] = (img.shape[1], img.shape[0])  # (W,H)
        
        # SIFT matching
        kptsA_xy, kptsB_xy = sift_match_keypoints(img, template, ratio=ratio)
        if kptsA_xy is None:
            return result
        
        # RANSAC homography
        H, inliers = ransac_homography(kptsA_xy, kptsB_xy, reproj_thresh=reproj_thresh)
        if H is None:
            return result
        
        num_inl = int(inliers.sum())
        result['num_inliers'] = num_inl
        
        if num_inl < min_inliers:
            return result
        
        # Inlier image points (pixel coords)
        img_inl = kptsA_xy[inliers, :]  # (N,2)
        
        # Inlier "template points" converted to pseudo-metric plane
        tpl_inl_pix = kptsB_xy[inliers, :]  # (N,2) in template pixels
        tpl_inl_plane = normalize_template_xy(tpl_inl_pix, template_shape_hw, template_size)
        
        result['success'] = True
        result['H'] = H
        result['img_inl'] = img_inl.reshape(-1, 1, 2).astype(np.float32)
        result['obj_inl'] = template_keypoints2objpoints(tpl_inl_plane)
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
    
    return result

# -----------------------
# Main pipeline with multiprocessing
# -----------------------
def rip_colmap_sift(img_paths, template_path, template_size=0.762,
                    ratio=0.75, reproj_thresh=3.0, min_inliers=30,
                    n_workers=None):
    """
    Main calibration pipeline with multiprocessing support.
    
    Args:
        img_paths: List of image paths
        template_path: Path to template image
        template_size: Physical size of template
        ratio: Lowe's ratio test threshold
        reproj_thresh: RANSAC reprojection threshold
        min_inliers: Minimum number of inliers required
        n_workers: Number of worker processes (None = use all CPUs)
    
    Returns:
        Hs, K, d, rvec, tvec, rep, good_imgs
    """
    img_paths = sorted(img_paths)
    
    # Load template to get its shape
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    template_shape_hw = template.shape[:2]
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Processing {len(img_paths)} images with {n_workers} workers...")
    
    # Create partial function with fixed parameters
    worker_func = partial(
        process_single_image,
        template_path=template_path,
        template_shape_hw=template_shape_hw,
        template_size=template_size,
        ratio=ratio,
        reproj_thresh=reproj_thresh,
        min_inliers=min_inliers
    )
    
    # Process images in parallel
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_func, img_paths),
            total=len(img_paths),
            desc="Processing images"
        ))
    
    # Collect successful results
    Hs = []
    img_pnts_all = []
    obj_pnts_all = []
    good_imgs = []
    last_img_shape = None
    
    for result in results:
        if result['success']:
            Hs.append(result['H'])
            img_pnts_all.append(result['img_inl'])
            obj_pnts_all.append(result['obj_inl'])
            good_imgs.append(result['img_path'])
            last_img_shape = result['img_shape']
            print(f"✓ {os.path.basename(result['img_path'])}: {result['num_inliers']} inliers")
        else:
            print(f"✗ {os.path.basename(result['img_path'])}: {result['num_inliers']} inliers (skipped)")
    
    if len(img_pnts_all) < 3:
        raise RuntimeError(f"Not enough valid views for calibration. Got {len(img_pnts_all)}, need at least 3.")
    
    print(f"\nCalibrating camera with {len(img_pnts_all)} views...")
    
    # Calibrate camera
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_ASPECT_RATIO
    rep, K, d, rvec, tvec = cv2.calibrateCamera(
        obj_pnts_all,
        img_pnts_all,
        last_img_shape,  # (W,H)
        None,
        None,
        flags=flags
    )
    
    print(f"Calibration complete! Reprojection error: {rep:.4f}")
    
    return Hs, K, d, rvec, tvec, rep, good_imgs

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    import argparse
    import PIL.Image
    
    parser = argparse.ArgumentParser(description="Camera calibration using SIFT matching with multiprocessing")
    parser.add_argument("--template_size", type=float, default=0.762, help="Physical size of template")
    parser.add_argument("--template_path", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "Template.png"),
                        help="Path to template image")
    parser.add_argument("--img_dir", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "ffmpeg_imgs"),
                        help="Directory containing images")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    parser.add_argument("--reproj_thresh", type=float, default=3.0, help="RANSAC reprojection threshold")
    parser.add_argument("--min_inliers", type=int, default=40, help="Minimum number of inliers")
    parser.add_argument("--out_dir", type=str, default="colmap_model", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of worker processes (default: auto)")
    parser.add_argument("--visualize", action="store_true", help="Show 3D camera visualization")
    
    args = parser.parse_args()
    
    from dataset_helpers import save_calibration_pickle, export_colmap_text_model
    
    # Get image paths
    img_paths = glob.glob(os.path.join(args.img_dir, "*.jpg"))
    if not img_paths:
        img_paths = glob.glob(os.path.join(args.img_dir, "*.png"))
    
    if not img_paths:
        raise ValueError(f"No images found in {args.img_dir}")
    
    print(f"Found {len(img_paths)} images")
    
    # Run calibration
    Hs, K, d, rvec, tvec, rep, good_imgs = rip_colmap_sift(
        img_paths,
        args.template_path,
        template_size=args.template_size,
        ratio=args.ratio,
        reproj_thresh=args.reproj_thresh,
        min_inliers=args.min_inliers,
        n_workers=args.n_workers,
    )
    
    print(f"\nResults:")
    print(f"  Valid images: {len(Hs)}/{len(img_paths)}")
    print(f"  Reprojection error: {rep:.6f}")
    print(f"\nIntrinsic K:\n{K}")
    print(f"\nDistortion d:\n{d}")
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    img_size_wh = PIL.Image.open(img_paths[0]).size
    
    save_calibration_pickle(Hs, K, d, rvec, tvec, rep, 
                           f"{args.out_dir}/calibration.pkl", good_imgs)
    
    export_colmap_text_model(args.out_dir, good_imgs, K, d, rvec, tvec, 
                           image_size_wh=img_size_wh, 
                           camera_model="FULL_OPENCV", 
                           copy_images=True)
    
    # Visualization (optional)
    if args.visualize:
        import matplotlib
        #matplotlib.use('Agg')  # Use non-interactive backend for SSH
        import matplotlib.pyplot as plt
        
        # Analyze camera positions for better plot limits
        cam_positions = []
        for i in range(len(rvec)):
            R_w2c = cv2.Rodrigues(rvec[i])[0]
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ tvec[i].reshape(3, 1)
            cam_positions.append(t_c2w.flatten())
        
        cam_positions = np.array(cam_positions)
        
        # Print camera position stats
        print("\nCamera Position Statistics:")
        print(f"  X: [{cam_positions[:, 0].min():.3f}, {cam_positions[:, 0].max():.3f}]")
        print(f"  Y: [{cam_positions[:, 1].min():.3f}, {cam_positions[:, 1].max():.3f}]")
        print(f"  Z: [{cam_positions[:, 2].min():.3f}, {cam_positions[:, 2].max():.3f}]")
        print(f"  Mean: [{cam_positions.mean(axis=0)[0]:.3f}, {cam_positions.mean(axis=0)[1]:.3f}, {cam_positions.mean(axis=0)[2]:.3f}]")
        
        # Set plot limits based on actual camera positions
        margin = 0.2
        x_range = cam_positions[:, 0].max() - cam_positions[:, 0].min()
        y_range = cam_positions[:, 1].max() - cam_positions[:, 1].min()
        z_range = cam_positions[:, 2].max() - cam_positions[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        
        center = cam_positions.mean(axis=0)
        
        fig_in = plt.figure(figsize=(12, 10))
        ax_in = fig_in.add_subplot(projection='3d')
        
        ax_in.set_xlim(center[0] - max_range/2 * (1+margin), center[0] + max_range/2 * (1+margin))
        ax_in.set_ylim(center[1] - max_range/2 * (1+margin), center[1] + max_range/2 * (1+margin))
        ax_in.set_zlim(center[2] - max_range/2 * (1+margin), center[2] + max_range/2 * (1+margin))
        ax_in.set_xlabel('X')
        ax_in.set_ylabel('Y')
        ax_in.set_zlabel('Z')
        ax_in.set_title(f'Camera Poses ({len(rvec)} cameras)\nReprojection Error: {rep:.4f}')
        
        # Plot origin
        ax_in.scatter([0], [0], [0], c='black', s=100, marker='o', label='Origin')
        
        # Plot cameras
        for i in range(len(rvec)):
            R_w2c = cv2.Rodrigues(rvec[i])[0]
            R_c2w = R_w2c.T
            t_c2w = -R_c2w.dot(tvec[i]).reshape((3, 1))
            plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=max_range*0.05)
        
        ax_in.legend()
        plotPlane(ax_in, [0, 0, 1], 0)
        plotPlane(ax_in, [0, 1, 0], args.template_size/2)
        plotPlane(ax_in, [0, 1, 0], -args.template_size/2)
        plotPlane(ax_in, [1, 0, 0], args.template_size/2)
        plotPlane(ax_in, [1, 0, 0], -args.template_size/2)
        plt.show()
        plt.savefig(f"{args.out_dir}/camera_poses.png", dpi=150, bbox_inches='tight')
        print(f"\nCamera poses visualization saved to {args.out_dir}/camera_poses.png")