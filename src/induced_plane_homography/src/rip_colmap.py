import os 
import glob
import cv2
import numpy as np
import torch
from romav2 import RoMaV2
from romav2.geometry import prec_mat_from_prec_params
from torch.nn import functional as F
from poselib import estimate_homography
from tqdm import tqdm
import gc
import pickle
romav2 = RoMaV2().cuda()
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
def get_tensors(img, model):
    img_tensor = (torch.from_numpy(img)/255.0).permute(2, 0, 1).unsqueeze(0).cuda()
    img_lr = F.interpolate(img_tensor, size=(model.H_lr, model.W_lr), mode='bicubic', align_corners=False, antialias=True)
    img_hr = F.interpolate(img_tensor, size=(model.H_hr,  model.W_hr), mode='bicubic', align_corners=False, antialias=True)
    return img_lr, img_hr
def _map_confidence(*, confidence: torch.Tensor, threshold: float | None):
    overlap = confidence[..., :1].sigmoid()
    if threshold is not None:
        overlap[overlap > threshold] = 1.0
    precision = prec_mat_from_prec_params(confidence[..., 1:4])
    return overlap, precision
def get_matches(img, template, t_lr=None, t_hr=None):
    H_t, W_t = template.shape[:2]
    H_i, W_i = img.shape[:2]
    img_lr, img_hr = get_tensors(img, romav2)
    if t_lr is None:
        template_lr, template_hr = get_tensors(template, romav2)
    else:
        template_lr, template_hr = t_lr, t_hr
    preds = romav2(img_lr, template_lr, img_hr, template_hr)
    del img_lr, img_hr
    return preds
def sample_matches(matches, img, template, threshold=0.5):
    H_A, W_A = img.shape[:2]
    H_B, W_B = template.shape[:2]
    overlap_ti, precision_ti = _map_confidence(confidence=matches["confidence_BA"], threshold=threshold)
    matches["overlap_BA"] = overlap_ti
    matches["precision_BA"] = precision_ti
    overlap_it, precision_it = _map_confidence(confidence=matches["confidence_AB"], threshold=threshold)
    matches["overlap_AB"] = overlap_it
    matches["precision_AB"] = precision_it
    matches, overlaps, precision_i2t, precision_t2i = romav2.sample(matches, 10000)
    kptsA, kptsB = romav2.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    return kptsA, kptsB, overlaps, precision_i2t, precision_t2i

def template_keypoints2objpoints(keypoints_template_xy: np.ndarray) -> np.ndarray:
    """
    Convert template 2D keypoints (x,y) into pseudo object points (x,y,0).
    NOTE: This is a planar assumption (Z=0), not a real 3D calibration target.
    """
    keypoints_template_xy = np.asarray(keypoints_template_xy, dtype=np.float32).reshape(-1, 2)
    z = np.zeros((keypoints_template_xy.shape[0], 1), dtype=np.float32)
    obj = np.hstack([keypoints_template_xy, z])            # (N,3)
    return obj.reshape(-1, 1, 3)    


def rip_colmap(img_paths, template_path, template_size=0.762):
    img_paths.sort()
    template = cv2.imread(template_path)
    Hs = []
    img_pnts_all = []
    obj_pnts_all = []
    for i , img_path in tqdm(enumerate(img_paths)):

        img = cv2.imread(img_path)
        with torch.no_grad():
            matches = get_matches(img, template)
            kptsA, kptsB, overlaps, precision_i2t, precision_t2i = sample_matches(matches, img, template, threshold=0.5)
            kptsB = kptsB.float().detach().cpu().numpy()
            H_B, W_B = template.shape[:2]
            kptsA = kptsA.float().detach().cpu().numpy()
            del matches, overlaps, precision_i2t, precision_t2i
            gc.collect()
            torch.cuda.empty_cache()
        H, ret = estimate_homography(kptsA, kptsB, ransac_opt={"progressive_sampling": True, "max_reproj_error": 3})
        Hs.append(H)
        scale = np.mean([H_B, W_B])
        obj_xy = kptsB[ret["inliers"], :]/scale - 0.5
        obj_pnts = template_size*obj_xy
        img_pnts_all.append(kptsA[ret["inliers"], :])
        obj_pnts_all.append(template_keypoints2objpoints(obj_pnts))
    # calibrate camera
    rep, K, d, rvec, tvec = cv2.calibrateCamera(obj_pnts_all, img_pnts_all, (img.shape[1], img.shape[0]), None, None, flags=cv2.CALIB_RATIONAL_MODEL+cv2.CALIB_FIX_ASPECT_RATIO)
    return Hs, K, d, rvec, tvec, rep


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset_helpers import save_calibration_pickle
    img_paths = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "ffmpeg_imgs", "*.jpg"))
    template_path = os.path.join(os.path.dirname(__file__), "..","data", "plant_4", "Template.png")
    Hs, K, d, rvec, tvec, rep = rip_colmap(img_paths, template_path)
    save_calibration_pickle(Hs, K, d, rvec, tvec, rep, os.path.join(os.path.dirname(__file__), "..", "data", "plant_4", "calibration.pkl"))
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