import cv2
import numpy as np
import os
import glob

# ── Configure these to match your printed checkerboard ──────────────────────
CHECKERBOARD = (5, 3)   # inner corners (width, height) — NOT the number of squares
SQUARE_SIZE_MM = 11.5   # measure one square on your printout with a ruler
# ────────────────────────────────────────────────────────────────────────────

IMAGE_DIR = "calibration_images"
OUTPUT_PATH = "data/calibration.npz"

# cv2.findChessboardCorners is unreliable on full-resolution (e.g. 4K) images — its
# internal quad-detection heuristics assume comparatively small squares, and on our
# images it only found the board in 2/20 cases at full size vs. 20/20 once downscaled.
# So detection runs on a shrunk copy, then the found corners are scaled back up and
# refined with cornerSubPix on the full-resolution image for accurate subpixel corners.
DETECT_MAX_WIDTH = 600


def build_object_points():
    """3D coordinates of checkerboard corners in the checkerboard's own plane (z=0)."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def detect_corners(images):
    objp = build_object_points()
    obj_points = []
    img_points = []
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in sorted(images):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        scale = min(1.0, DETECT_MAX_WIDTH / gray.shape[1])
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
        found, corners = cv2.findChessboardCorners(small, CHECKERBOARD, None)
        if found:
            if scale < 1.0:
                corners /= scale
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            print(f"  OK   {os.path.basename(path)}")
        else:
            print(f"  SKIP {os.path.basename(path)}  (corners not found)")

    return obj_points, img_points, img_shape


def main():
    images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not images:
        print(f"No images found in '{IMAGE_DIR}/'.")
        return

    print(f"Found {len(images)} images. Detecting corners...")
    obj_points, img_points, img_shape = detect_corners(images)
    good = len(obj_points)
    print(f"\n{good}/{len(images)} images usable.")

    if good < 6:
        print("Need at least 6 good images — capture more and retry.")
        return

    print("Running calibration...")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    os.makedirs("data", exist_ok=True)
    np.savez(OUTPUT_PATH, K=K, dist=dist)

    print(f"\nCamera matrix (K):\n{K}")
    print(f"\nDistortion coefficients:\n{dist.ravel()}")
    print(f"\nRMS reprojection error: {rms:.4f} px  (good if < 1.0)")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
