import cv2
import numpy as np
from scipy.interpolate import CubicSpline

CAMERA_INDEX = 4
DISPLAY_SIZE = (3840, 2160)  # must match DISPLAY_SIZE in compute_homography.py
CALIBRATION_PATH = "data/calibration.npz"
HOMOGRAPHY_PATH = "data/homography.npz"

# Region of the BEV image shown for measurement, as fractions of bev_size.
# Adjust these to frame the cable tightly.
CROP_X = (0.4, 0.6)   # horizontal: centre half
CROP_Y = (0.4, 0.8)   # vertical:   lower half

N_SAMPLES = 1000   # points to sample along each spline for drawing and measurement

EDGE_COLORS = [(0, 220, 0), (0, 60, 255)]   # BGR: green, red
EDGE_NAMES = ["edge 1", "edge 2"]


# ── Spline ───────────────────────────────────────────────────────────────────

def fit_spline(points):
    """Fit a parametric cubic spline through clicked points.
    Returns an (N, 2) array of sampled (x, y) positions, or None if < 2 points."""
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=float)
    t = np.arange(len(pts))
    t_fine = np.linspace(0, len(pts) - 1, N_SAMPLES)
    if len(pts) == 2:
        xs = np.interp(t_fine, t, pts[:, 0])
        ys = np.interp(t_fine, t, pts[:, 1])
    else:
        xs = CubicSpline(t, pts[:, 0])(t_fine)
        ys = CubicSpline(t, pts[:, 1])(t_fine)
    return np.stack([xs, ys], axis=1)


def compute_diameter(spline1, spline2, pixels_per_mm):
    """Mean horizontal distance between splines over their shared y-range, in mm.
    Only the overlap region is used so unequal spline lengths don't inflate the result.
    For a near-vertical cable this equals the orthogonal distance to within ~3.5%."""
    def sorted_by_y(spline):
        order = np.argsort(spline[:, 1])
        return spline[order, 1], spline[order, 0]   # ys, xs

    ys1, xs1 = sorted_by_y(spline1)
    ys2, xs2 = sorted_by_y(spline2)

    y_min = max(ys1[0], ys2[0])
    y_max = min(ys1[-1], ys2[-1])
    if y_min >= y_max:
        return None   # splines don't overlap vertically

    y_samples = np.linspace(y_min, y_max, N_SAMPLES)
    x1 = np.interp(y_samples, ys1, xs1)
    x2 = np.interp(y_samples, ys2, xs2)

    return float(np.mean(np.abs(x1 - x2))) / pixels_per_mm


# ── Drawing ──────────────────────────────────────────────────────────────────

def put_text(img, text, pos, scale=0.7, thickness=2):
    """White text with black outline for readability on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def draw_overlay(base, points, active_edge, splines, diameter_mm):
    img = base.copy()

    # Splines
    for i, spline in enumerate(splines):
        if spline is not None:
            pts = spline.astype(np.int32)
            for j in range(len(pts) - 1):
                cv2.line(img, tuple(pts[j]), tuple(pts[j + 1]), EDGE_COLORS[i], 2)

    # Clicked points
    for i, edge_pts in enumerate(points):
        for pt in edge_pts:
            cv2.circle(img, pt, 5, EDGE_COLORS[i], -1)
            cv2.circle(img, pt, 5, (255, 255, 255), 1)

    # Result
    if diameter_mm is not None:
        put_text(img, f"Diameter: {diameter_mm:.2f} mm", (15, 40), scale=1.1, thickness=2)

    # Instructions
    active_color = EDGE_COLORS[active_edge]
    label = f"Active: {EDGE_NAMES[active_edge]}"
    cv2.putText(img, label, (14, img.shape[0] - 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(img, label, (14, img.shape[0] - 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, active_color, 2)
    put_text(img, "TAB=switch edge  z=undo  r=reset  ENTER=calculate  q=quit",
             (14, img.shape[0] - 24), scale=0.6)

    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cal = np.load(CALIBRATION_PATH)
    K, dist = cal["K"], cal["dist"]

    hom = np.load(HOMOGRAPHY_PATH)
    H = hom["H"]
    bev_size = (int(hom["output_w"]), int(hom["output_h"]))
    pixels_per_mm = float(hom["pixels_per_mm"])

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    print("Press SPACE to freeze the BEV frame for measurement.")

    # Pixel bounds of the crop region within the BEV image
    cx1 = int(bev_size[0] * CROP_X[0])
    cx2 = int(bev_size[0] * CROP_X[1])
    cy1 = int(bev_size[1] * CROP_Y[0])
    cy2 = int(bev_size[1] * CROP_Y[1])
    crop_size = (cx2 - cx1, cy2 - cy1)

    cv2.namedWindow("Live BEV - SPACE to freeze", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live BEV - SPACE to freeze", *bev_size)

    frozen = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.rotate(f, cv2.ROTATE_180)
        undistorted = cv2.resize(cv2.undistort(f, K, dist), DISPLAY_SIZE)
        warped = cv2.warpPerspective(undistorted, H, bev_size)
        cv2.imshow("Live BEV - SPACE to freeze", warped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frozen = warped[cy1:cy2, cx1:cx2].copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if frozen is None:
        return

    points = [[], []]
    active_edge = 0
    diameter_mm = None

    def on_mouse(event, x, y, flags, param):
        nonlocal diameter_mm
        if event == cv2.EVENT_LBUTTONDOWN:
            points[active_edge].append((x, y))
            diameter_mm = None

    cv2.namedWindow("Measure diameter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Measure diameter", *crop_size)
    cv2.setMouseCallback("Measure diameter", on_mouse)

    print("Click along each cable edge to place spline points.")
    print("TAB = switch edge | z = undo | r = reset | ENTER = calculate | q = quit")

    while True:
        splines = [fit_spline(points[0]), fit_spline(points[1])]
        img = draw_overlay(frozen, points, active_edge, splines, diameter_mm)
        cv2.imshow("Measure diameter", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 9:   # TAB
            active_edge = 1 - active_edge
        elif key == ord("z"):
            if points[active_edge]:
                points[active_edge].pop()
                diameter_mm = None
        elif key == ord("r"):
            points = [[], []]
            diameter_mm = None
        elif key == 13:  # ENTER
            if splines[0] is not None and splines[1] is not None:
                diameter_mm = compute_diameter(splines[0], splines[1], pixels_per_mm)
                print(f"Diameter: {diameter_mm:.2f} mm")
            else:
                print("Place at least 2 points on each edge first.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
