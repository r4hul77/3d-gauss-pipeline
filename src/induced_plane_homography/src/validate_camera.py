#!/usr/bin/env python3
"""
Validate that camera setup looks reasonable for object scanning.
"""
import sys
import pickle
import numpy as np
import cv2

def validate_cameras(pkl_path):
    print("=" * 70)
    print("CAMERA SETUP VALIDATION")
    print("=" * 70)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    rvecs = data['rvec']
    tvecs = data['tvec']
    rep_error = data['rep']
    
    # Compute camera positions
    camera_positions = []
    viewing_angles_to_origin = []
    
    for rvec, tvec in zip(rvecs, tvecs):
        R_w2c, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        C = -R_w2c.T @ tvec.reshape(3, 1)
        camera_positions.append(C.flatten())
        
        # Camera forward direction
        forward_cam = np.array([0, 0, -1])
        forward_world = R_w2c.T @ forward_cam
        
        # Direction to origin from camera
        to_origin = -C.flatten() / np.linalg.norm(C)
        
        # Angle between viewing direction and to-origin
        dot = np.dot(forward_world, to_origin)
        angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
        viewing_angles_to_origin.append(angle)
    
    camera_positions = np.array(camera_positions)
    viewing_angles_to_origin = np.array(viewing_angles_to_origin)
    
    # Statistics
    distances = np.linalg.norm(camera_positions, axis=1)
    
    print(f"\n✓ Number of cameras: {len(camera_positions)}")
    print(f"✓ Reprojection error: {rep_error:.4f} pixels")
    
    print(f"\n📊 Camera Distances from Origin:")
    print(f"   Mean: {distances.mean():.4f} ± {distances.std():.4f}")
    print(f"   Range: [{distances.min():.4f}, {distances.max():.4f}]")
    
    print(f"\n📐 Viewing Angles (angle to origin):")
    print(f"   Mean: {viewing_angles_to_origin.mean():.1f}° ± {viewing_angles_to_origin.std():.1f}°")
    print(f"   Range: [{viewing_angles_to_origin.min():.1f}°, {viewing_angles_to_origin.max():.1f}°]")
    
    # Validation checks
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Reprojection error
    total_checks += 1
    if rep_error < 1.0:
        print("✅ Excellent reprojection error (< 1.0 pixel)")
        checks_passed += 1
    elif rep_error < 2.0:
        print("✅ Good reprojection error (< 2.0 pixels)")
        checks_passed += 1
    elif rep_error < 5.0:
        print("⚠️  Moderate reprojection error (2-5 pixels)")
    else:
        print("❌ High reprojection error (> 5 pixels) - calibration may be poor")
    
    # Check 2: Cameras roughly equidistant
    total_checks += 1
    distance_cv = distances.std() / distances.mean()  # Coefficient of variation
    if distance_cv < 0.1:
        print(f"✅ Cameras at consistent distances (CV: {distance_cv:.3f})")
        checks_passed += 1
    elif distance_cv < 0.3:
        print(f"⚠️  Some variation in camera distances (CV: {distance_cv:.3f})")
    else:
        print(f"❌ Large variation in camera distances (CV: {distance_cv:.3f})")
    
    # Check 3: Cameras looking at origin
    total_checks += 1
    looking_at_origin = np.sum(viewing_angles_to_origin < 30)
    if looking_at_origin > len(camera_positions) * 0.9:
        print(f"✅ Most cameras looking at origin ({looking_at_origin}/{len(camera_positions)})")
        checks_passed += 1
    elif looking_at_origin > len(camera_positions) * 0.7:
        print(f"⚠️  Many cameras looking at origin ({looking_at_origin}/{len(camera_positions)})")
    else:
        print(f"❌ Few cameras looking at origin ({looking_at_origin}/{len(camera_positions)})")
    
    # Check 4: Camera coverage
    total_checks += 1
    # Check azimuthal distribution
    azimuth = np.arctan2(camera_positions[:, 1], camera_positions[:, 0])
    azimuth_sorted = np.sort(azimuth)
    azimuth_gaps = np.diff(azimuth_sorted)
    max_gap = np.max(azimuth_gaps) * 180 / np.pi
    
    if max_gap < 30:
        print(f"✅ Good azimuthal coverage (max gap: {max_gap:.1f}°)")
        checks_passed += 1
    elif max_gap < 60:
        print(f"⚠️  Moderate azimuthal coverage (max gap: {max_gap:.1f}°)")
    else:
        print(f"❌ Poor azimuthal coverage (max gap: {max_gap:.1f}°)")
    
    # Overall assessment
    print("\n" + "=" * 70)
    print(f"OVERALL: {checks_passed}/{total_checks} checks passed")
    print("=" * 70)
    
    if checks_passed == total_checks:
        print("\n🎉 Excellent! Camera setup looks great for 3D reconstruction.")
        print("   Ready for Gaussian splatting or NeRF training.")
    elif checks_passed >= total_checks * 0.75:
        print("\n👍 Good camera setup. Should work well for 3D reconstruction.")
    elif checks_passed >= total_checks * 0.5:
        print("\n⚠️  Acceptable camera setup, but could be improved.")
    else:
        print("\n❌ Camera setup has issues. Consider recalibrating.")
    
    # Specific recommendations
    print("\n📋 Recommendations:")
    if rep_error > 2.0:
        print("   - Lower --ratio and increase --min_inliers for better matching")
    if distance_cv > 0.2:
        print("   - Check if some images have poor matches or different zoom levels")
    if looking_at_origin < len(camera_positions) * 0.8:
        print("   - Verify template is centered and visible in all images")
    if max_gap > 45:
        print("   - Add more images to fill coverage gaps")
    
    if checks_passed == total_checks and rep_error < 1.0:
        print("   - None! This calibration looks excellent. 🚀")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_camera_setup.py calibration.pkl")
        sys.exit(1)
    
    validate_cameras(sys.argv[1])