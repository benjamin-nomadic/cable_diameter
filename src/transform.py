import cv2
import numpy as np


def load(calibration_path, homography_path):
    """Load camera calibration and BEV homography from disk.

    Returns (K, dist, H, bev_size, pixels_per_mm).
    bev_size is (width, height) in pixels.
    """
    cal = np.load(calibration_path)
    hom = np.load(homography_path)
    return (
        cal["K"],
        cal["dist"],
        hom["H"],
        (int(hom["output_w"]), int(hom["output_h"])),
        float(hom["pixels_per_mm"]),
    )


def apply(image, K, dist, H, bev_size, display_size, crop_x=(0.0, 1.0), crop_y=(0.0, 1.0)):
    """Undistort, warp to bird's-eye view, and crop to the region of interest.

    display_size: (w, h) resolution the homography was computed in.
    crop_x, crop_y: (low_fraction, high_fraction) of bev_size to keep.
    Returns the cropped BEV image.
    """
    undistorted = cv2.resize(cv2.undistort(image, K, dist), display_size)
    warped = cv2.warpPerspective(undistorted, H, bev_size)

    cx1, cx2 = int(bev_size[0] * crop_x[0]), int(bev_size[0] * crop_x[1])
    cy1, cy2 = int(bev_size[1] * crop_y[0]), int(bev_size[1] * crop_y[1])
    return warped[cy1:cy2, cx1:cx2].copy()
