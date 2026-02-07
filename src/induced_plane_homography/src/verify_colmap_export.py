#!/usr/bin/env python3
"""
Verify that COLMAP export matches the calibration.
Load both the pickle and the exported COLMAP model and compare.
"""

import sys
import pickle
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

def rotmat_to_qvec(R):
    """Convert rotation matrix to COLMAP quaternion [w, x, y, z]"""
    r = Rot.from_matrix(R)
    quat = r.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]

def read_colmap_images(images_txt_path):
    """Read COLMAP images.txt file"""
    images = {}
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments
    lines = [l for l in lines if not l.startswith('#')]
    
    # Parse images (every 2 lines)
    for i in range(0, len(lines), 2):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) < 10:
            continue
        
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        
        images[name] = {
            'id': image_id,
            'qvec': np.array([qw, qx, qy, qz]),
            'tvec': np.array([tx, ty, tz]),
            'camera_id': camera_id
        }
    
    return images

def compare_calibration_to_colmap(pkl_path, colmap_dir):
    """Compare pickle calibration to exported COLMAP model"""
    
    print("=" * 70)
    print("CALIBRATION vs COLMAP EXPORT VERIFICATION")
    print("=" * 70)
    
    # Load pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    rvecs = data['rvec']
    tvecs = data['tvec']
    good_imgs = data['good_imgs']
    
    # Load COLMAP
    colmap_path = Path(colmap_dir)
    images_txt = colmap_path / 'sparse' / '0' / 'images.txt'
    
    if not images_txt.exists():
        print(f"❌ COLMAP images.txt not found at {images_txt}")
        return
    
    colmap_images = read_colmap_images(images_txt)
    
    print(f"\n📁 Files:")
    print(f"   Pickle: {pkl_path}")
    print(f"   COLMAP: {images_txt}")
    
    print(f"\n📊 Counts:")
    print(f"   Calibration images: {len(good_imgs)}")
    print(f"   COLMAP images: {len(colmap_images)}")
    
    # Compare each image
    print("\n" + "=" * 70)
    print("POSE COMPARISON (First 5 images)")
    print("=" * 70)
    
    max_quat_error = 0
    max_tvec_error = 0
    
    for i, (rvec, tvec, img_path) in enumerate(zip(rvecs, tvecs, good_imgs)):
        img_name = Path(img_path).name
        
        if img_name not in colmap_images:
            print(f"⚠️  {img_name} not found in COLMAP export")
            continue
        
        # Convert pickle pose to COLMAP format
        R_w2c, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        R_c2w = R_w2c.T
        tvec_w2c = tvec.reshape(3, 1)
        tvec_c2w = -R_c2w @ tvec_w2c
        
        qvec_expected = rotmat_to_qvec(R_c2w)
        tvec_expected = tvec_c2w.flatten()
        
        # Get COLMAP pose
        colmap_img = colmap_images[img_name]
        qvec_colmap = colmap_img['qvec']
        tvec_colmap = colmap_img['tvec']
        
        # Compare
        quat_error = np.linalg.norm(qvec_expected - qvec_colmap)
        tvec_error = np.linalg.norm(tvec_expected - tvec_colmap)
        
        max_quat_error = max(max_quat_error, quat_error)
        max_tvec_error = max(max_tvec_error, tvec_error)
        
        if i < 5:  # Print first 5
            print(f"\n{img_name}:")
            print(f"  Quaternion:")
            print(f"    Expected: [{qvec_expected[0]:.6f}, {qvec_expected[1]:.6f}, {qvec_expected[2]:.6f}, {qvec_expected[3]:.6f}]")
            print(f"    COLMAP:   [{qvec_colmap[0]:.6f}, {qvec_colmap[1]:.6f}, {qvec_colmap[2]:.6f}, {qvec_colmap[3]:.6f}]")
            print(f"    Error: {quat_error:.2e}")
            
            print(f"  Translation:")
            print(f"    Expected: [{tvec_expected[0]:8.4f}, {tvec_expected[1]:8.4f}, {tvec_expected[2]:8.4f}]")
            print(f"    COLMAP:   [{tvec_colmap[0]:8.4f}, {tvec_colmap[1]:8.4f}, {tvec_colmap[2]:8.4f}]")
            print(f"    Error: {tvec_error:.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nMaximum errors:")
    print(f"  Quaternion: {max_quat_error:.2e}")
    print(f"  Translation: {max_tvec_error:.2e}")
    
    if max_quat_error < 1e-6 and max_tvec_error < 1e-6:
        print("\n✅ PERFECT MATCH! Export is correct.")
    elif max_quat_error < 1e-3 and max_tvec_error < 1e-3:
        print("\n✅ Very close match (within numerical precision)")
    elif max_quat_error < 0.1 and max_tvec_error < 0.1:
        print("\n⚠️  Small differences detected")
    else:
        print("\n❌ Large differences detected - export may be wrong!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_colmap_export.py calibration.pkl colmap_output_dir/")
        sys.exit(1)
    
    compare_calibration_to_colmap(sys.argv[1], sys.argv[2])