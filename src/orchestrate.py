import sys
import cv2

import transform
import edge_proposal
import manual_refinement
import calculate_diameter

CALIBRATION_PATH = "data/calibration.npz"
HOMOGRAPHY_PATH  = "data/homography.npz"
DISPLAY_SIZE     = (3840, 2160)   # resolution the homography was computed in
CAMERA_INDEX     = 4
CROP_X           = (0.3, 0.7)    # horizontal slice of the BEV image to measure
CROP_Y           = (0.25, 0.6)    # vertical slice of the BEV image to measure
HEIGHT_MM        = 48           # set to camera height above cable plane in mm to enable geometric correction

# ── Edge detection parameters (see edge_proposal.propose_test) ──────────────────
EDGE_WEIGHTS           = (1.0, 0.5, 0.5)   # (L, a, b) channel weights
EDGE_BLUR_SIGMA        = 7
EDGE_SOBEL_KSIZE       = 3
EDGE_THRESHOLD_PCT     = 50
EDGE_MIN_LINE_SPAN_PCT = 20
EDGE_MIN_SEP_MM        = 5


def _capture_from_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    win = "Live — SPACE to capture  Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    print("Press SPACE to capture a frame.")

    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.rotate(f, cv2.ROTATE_180)
        cv2.imshow(win, cv2.resize(f, (1280, 720)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame = f
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def main():
    # ── 1. Acquire image ──────────────────────────────────────────────────────
    if len(sys.argv) >= 2:
        image = cv2.imread(sys.argv[1])
        if image is None:
            print(f"Cannot read image: {sys.argv[1]}")
            return
        print(f"Loaded {sys.argv[1]}")
    else:
        image = _capture_from_camera()
        if image is None:
            return

    # ── 2. Transform: undistort → BEV → crop ─────────────────────────────────
    K, dist, H, bev_size, pixels_per_mm = transform.load(CALIBRATION_PATH, HOMOGRAPHY_PATH)
    roi = transform.apply(image, K, dist, H, bev_size, DISPLAY_SIZE, CROP_X, CROP_Y)

    # ── 3. Propose edges ──────────────────────────────────────────────────────
    handles = edge_proposal.propose_test(
        roi, pixels_per_mm,
        weights=EDGE_WEIGHTS,
        blur_sigma=EDGE_BLUR_SIGMA,
        sobel_ksize=EDGE_SOBEL_KSIZE,
        threshold_pct=EDGE_THRESHOLD_PCT,
        min_line_span_pct=EDGE_MIN_LINE_SPAN_PCT,
        min_sep_mm=EDGE_MIN_SEP_MM,
    )
    if handles is None:
        print("No sensible edges detected — adjust the EDGE_* constants at the top of this file and try again.")
        return

    # ── 4. Manual refinement ──────────────────────────────────────────────────
    handles = manual_refinement.refine(
        roi, handles,
        compute_diameter=lambda h: calculate_diameter.calculate(h, pixels_per_mm, HEIGHT_MM),
    )

    # ── 5. Calculate and report diameter ──────────────────────────────────────
    diameter_mm = calculate_diameter.calculate(handles, pixels_per_mm, HEIGHT_MM)
    if diameter_mm is not None:
        print(f"\nDiameter: {diameter_mm:.2f} mm")
    else:
        print("Could not calculate diameter — ensure both edges were detected.")


if __name__ == "__main__":
    main()
