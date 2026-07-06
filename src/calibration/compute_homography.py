import cv2
import numpy as np
import os

CAMERA_INDEX = 4
CAMERA_RESOLUTION = (3840, 2160)  # must match the resolution used during calibration
DISPLAY_SIZE = (3840, 2160)       # must stay equal to CAMERA_RESOLUTION
CALIBRATION_PATH = "data/calibration.npz"
OUTPUT_PATH = "data/homography.npz"

# ── Measure your reference rectangle carefully with a ruler ──────────────────
REF_WIDTH_MM = 144
REF_HEIGHT_MM = 112
PIXELS_PER_MM = 5.0
PADDING_MM = 150.0   # extra scene shown around the reference rectangle on every side
# ─────────────────────────────────────────────────────────────────────────────

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]


def draw_state(img, points):
    display = img.copy()
    for i, pt in enumerate(points):
        cv2.circle(display, pt, 8, (0, 255, 0), -1)
        cv2.putText(display, CORNER_LABELS[i], (pt[0] + 12, pt[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) == 4:
        cv2.polylines(display, [np.array(points, np.int32)], True, (0, 255, 0), 2)
        msg = "Press ENTER to confirm  |  r = reset"
    else:
        msg = f"Click the {CORNER_LABELS[len(points)]} corner of your reference rectangle"
    cv2.putText(display, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return display


def capture_frame(K, dist):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    cv2.namedWindow("Live feed - press SPACE to freeze", cv2.WINDOW_AUTOSIZE)

    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.rotate(f, cv2.ROTATE_180)
        # Undistort and resize to DISPLAY_SIZE — this is the coordinate space we work in
        preview = cv2.resize(cv2.undistort(f, K, dist), DISPLAY_SIZE)
        cv2.imshow("Live feed - press SPACE to freeze", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame = preview.copy()
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def select_corners(frame):
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow("Select corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Select corners", on_mouse)

    while True:
        cv2.imshow("Select corners", draw_state(frame, points))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == 13 and len(points) == 4:  # ENTER
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return points  # in DISPLAY_SIZE coordinate space


def main():
    data = np.load(CALIBRATION_PATH)
    K, dist = data["K"], data["dist"]

    print("Place your reference rectangle flat in the scene.")
    print("Press SPACE to freeze the frame when ready.")
    frame = capture_frame(K, dist)
    if frame is None:
        return

    print("\nClick the 4 corners of your rectangle in this order:")
    print("  top-left -> top-right -> bottom-right -> bottom-left")
    print("Press ENTER to confirm, 'r' to reset.")
    points = select_corners(frame)
    if points is None:
        return

    ref_w = round(REF_WIDTH_MM * PIXELS_PER_MM)
    ref_h = round(REF_HEIGHT_MM * PIXELS_PER_MM)
    pad   = round(PADDING_MM   * PIXELS_PER_MM)

    # Place the reference rectangle at (pad, pad) inside a larger canvas.
    # Everything within PADDING_MM of the rectangle is also visible.
    dst = np.array([
        [pad,         pad        ],
        [pad + ref_w, pad        ],
        [pad + ref_w, pad + ref_h],
        [pad,         pad + ref_h],
    ], dtype=np.float32)

    full_w = ref_w + 2 * pad
    full_h = ref_h + 2 * pad

    src = np.array(points, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    flat = cv2.warpPerspective(frame, H, (full_w, full_h))
    cv2.namedWindow("BEV preview - press any key to save", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV preview - press any key to save", 800, round(800 * full_h / full_w))
    cv2.imshow("BEV preview - press any key to save", flat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    os.makedirs("data", exist_ok=True)
    np.savez(OUTPUT_PATH, H=H, output_w=np.int32(full_w), output_h=np.int32(full_h),
             pixels_per_mm=np.float64(PIXELS_PER_MM))
    print(f"\nSaved to {OUTPUT_PATH}  ({full_w}x{full_h}px at {PIXELS_PER_MM}px/mm)")


if __name__ == "__main__":
    main()
