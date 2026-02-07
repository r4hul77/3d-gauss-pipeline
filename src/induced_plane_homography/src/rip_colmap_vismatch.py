import os 
import glob
import cv2
import numpy as np
import torch
from vismatch import get_matcher
from poselib import estimate_homography
from tqdm import tqdm
import gc
import pickle
from PIL import Image

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


def template_keypoints2objpoints(template: Image.Image, keypoints_template_xy: np.ndarray, template_size: float):
    W, H = template.size
    normalized_kp = 2*keypoints_template_xy / (H + W) - 0.5
    return normalized_kp*template_size


def rip_colmap(img_paths, template_path, template_size=0.762, matcher_name=None):
    img_paths.sort()
    template = Image.open(template_path).convert("RGB")
    Hs = []
    Hm = []
    img_pnts_all = []
    obj_pnts_all = []
    img_paths_saved = []
    if matcher_name is None:
        matcher = get_matcher("romav2", device="cuda")
    else:
        matcher = get_matcher(matcher_name, device="cuda")
    for i , img_path in tqdm(enumerate(img_paths)):
        img = Image.open(img_path)
        img_paths_saved.append(img_path)
        with torch.no_grad():
            matches = matcher(img, template)
            kp_i = matches["inlier_kpts0"]
            kp_t = matches["inlier_kpts1"]
            kp_t = template_keypoints2objpoints(template, kp_t, template_size)
            gc.collect()
            torch.cuda.empty_cache()
        H, ret = estimate_homography(kp_i, kp_t, ransac_opt={"progressive_sampling": True, "max_reproj_error": 0.25})
        Hs.append(H)
        Hm.append(matches["H"])
        img_pnts_all.append(kp_i.astype(np.float32))
        obj_pnts_all.append(np.hstack([kp_t, np.zeros((kp_t.shape[0], 1))]).astype(np.float32))
    # calibrate camera
    rep, K, d, rvec, tvec = cv2.calibrateCamera(obj_pnts_all, img_pnts_all, (img.size[0], img.size[1]), None, None, flags=cv2.CALIB_THIN_PRISM_MODEL+cv2.CALIB_SAME_FOCAL_LENGTH )
    return Hs, K, d, rvec, tvec, rep, Hm, img_paths_saved

def save_calibration(Hs, K, d, rvec, tvec, rep, Hms, img_paths_saved, file_path):
    data = {
        "Hs": Hs,
        "K": K,
        "d": d,
        "rvec": rvec,
        "tvec": tvec,
        "rep": rep,
        "Hms": Hms,
        "img_paths": img_paths_saved
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Calibration saved to {file_path}")
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_paths = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "ffmpeg_imgs", "*.jpg"))
    template_path = os.path.join(os.path.dirname(__file__), "..","data", "plant_4", "Template.png")
    Hs, K, d, rvec, tvec, rep, Hms, img_paths_saved = rip_colmap(img_paths, template_path, matcher_name="romav2")
    save_calibration(Hs, K, d, rvec, tvec, rep, Hms, img_paths_saved, os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "calibration.pkl"))
    print("Reprojection error: ", rep)
    print("Intrinsic parameter K: ", K)
    print("Distortion parameters: ", d)
    fig_in = plt.figure()
    ax_in = fig_in.add_subplot(projection='3d')

    ax_in.set_xlim(-1, 1)
    ax_in.set_ylim(-1, 1)
    ax_in.set_zlim(-1, 1)
    
    for i in range(len(rvec)):
        R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i])[0])
        t_c2w = -R_c2w.dot(tvec[i]).reshape((1,3))
        plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=0.01)
    plt.show()