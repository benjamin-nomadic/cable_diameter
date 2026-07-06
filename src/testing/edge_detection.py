import sys
import cv2
import numpy as np

MIN_SEPARATION  = 40   # minimum rho gap (px) between the two selected lines
BIN_HALF_WIDTH  = 3    # pixels either side of a Hough line to highlight in the overlay
N_SAMPLES       = 500  # y-samples for diameter integration
DRAG_RADIUS     = 20   # px radius for draggable handle hit-test and circle drawing
HOMOGRAPHY_PATH = "data/homography.npz"


# ── 1. Preprocessing ──────────────────────────────────────────────────────────
# Convert the image to LAB and optionally blur each channel.
# This runs once on load (LAB split) and once per slider change (blur).

def to_lab_channels(bgr):
    """Split a BGR image into its L, a, b channels."""
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))


def blur_channels(channels, sigma):
    """Gaussian-blur each channel with the given sigma. sigma=0 is a no-op."""
    if sigma == 0:
        return channels
    return tuple(cv2.GaussianBlur(ch, (0, 0), sigma) for ch in channels)


# ── 2. Gradient calculation ───────────────────────────────────────────────────
# Compute the horizontal Sobel gradient on each LAB channel, normalise each to
# [0, 1] so the weight sliders are directly comparable, then combine.

def channel_gradients(channels, ksize):
    """Absolute horizontal Sobel gradient for each channel."""
    return tuple(
        np.abs(cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=ksize))
        for ch in channels
    )


def normalise_gradients(gradients):
    """Scale each gradient map to [0, 1] independently.
    Without this, L always dominates because it spans a wider value range than a/b."""
    return tuple(g / (g.max() or 1.0) for g in gradients)


def combine_gradients(norm_grads, weights):
    """Weighted sum of the normalised gradient maps."""
    return sum(w * g for w, g in zip(weights, norm_grads))


# ── 3. Line detection ─────────────────────────────────────────────────────────
# Threshold the combined gradient into a binary mask, then use the Hough
# transform to find the two strongest near-vertical lines.

def threshold_gradient(combined, threshold_pct):
    """Return a binary mask keeping pixels at or above threshold_pct % of the max."""
    combined_max = float(combined.max()) or 1.0
    return combined >= (threshold_pct / 100.0) * combined_max


def find_best_lines(mask, min_votes):
    """Hough-detect near-vertical lines (±15° of vertical) and return the two
    with the most support that are at least MIN_SEPARATION apart in rho."""
    lines = cv2.HoughLines(mask.astype(np.uint8) * 255,
                           rho=1, theta=np.pi / 900, threshold=min_votes)
    if lines is None:
        return None, None

    near_vert = [
        (float(r), float(t))
        for r, t in lines[:, 0]
        if t < np.pi / 12 or t > 11 * np.pi / 12
    ]
    if not near_vert:
        return None, None

    line1 = near_vert[0]
    line2 = None
    for rho, theta in near_vert[1:]:
        if abs(rho - line1[0]) > MIN_SEPARATION:
            line2 = (rho, theta)
            break

    return line1, line2


# ── 4. Visualisation ──────────────────────────────────────────────────────────

def line_endpoints(line, h, w):
    """Return two points on the line at y=0 and y=h-1 (x clipped to image bounds).
    These are the furthest-apart points the line has inside the image."""
    rho, theta = line
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if abs(cos_t) < 1e-6:
        return (int(rho), 0), (int(rho), h - 1)
    x_top = int(np.clip(rho / cos_t, 0, w - 1))
    x_bot = int(np.clip((rho - (h - 1) * sin_t) / cos_t, 0, w - 1))
    return (x_top, 0), (x_bot, h - 1)


def make_overlay(bgr, mask, line1, line2):
    """Colour the mask pixels near each detected line."""
    overlay = bgr.copy()
    ys, xs = np.where(mask)
    for line, colour in [(line1, (0, 220, 0)), (line2, (0, 60, 255))]:
        if line is None:
            continue
        rho, theta = line
        dists = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
        overlay[ys[dists <= BIN_HALF_WIDTH], xs[dists <= BIN_HALF_WIDTH]] = colour
    return overlay


def draw_handles(bgr, handles):
    """Draw the two edge lines and their draggable endpoint circles.

    handles: [[top1, bot1], [top2, bot2]] where each point is a [x, y] list,
             or None for a line that wasn't detected.
    """
    out = bgr.copy()
    if handles is None:
        return out
    colours = [(0, 220, 0), (0, 60, 255)]
    for i, pts in enumerate(handles):
        if pts is None:
            continue
        colour = colours[i]
        pt0 = (int(pts[0][0]), int(pts[0][1]))
        pt1 = (int(pts[1][0]), int(pts[1][1]))
        cv2.line(out, pt0, pt1, colour, 2)
        for pt in (pt0, pt1):
            cv2.circle(out, pt, DRAG_RADIUS, colour, 2)
            cv2.circle(out, pt, DRAG_RADIUS, (255, 255, 255), 1)
    return out


# ── 5. Diameter ───────────────────────────────────────────────────────────────

def compute_diameter_from_pts(handles, pixels_per_mm):
    """Mean horizontal distance between two line segments, in mm.

    handles: [[top0, bot0], [top1, bot1]] — each point is a [x, y] list.
    Uses the shared y-range of the two segments so unequal spans don't inflate the result.
    """
    if handles is None or handles[0] is None or handles[1] is None:
        return None

    def interp_x(pts, y):
        (xt, yt), (xb, yb) = pts
        if yb == yt:
            return float(xt)
        return xt + (xb - xt) * (y - yt) / (yb - yt)

    y0_min = min(handles[0][0][1], handles[0][1][1])
    y0_max = max(handles[0][0][1], handles[0][1][1])
    y1_min = min(handles[1][0][1], handles[1][1][1])
    y1_max = max(handles[1][0][1], handles[1][1][1])

    y_min = max(y0_min, y1_min)
    y_max = min(y0_max, y1_max)
    if y_min >= y_max:
        return None

    ys = np.linspace(y_min, y_max, N_SAMPLES)
    x0 = np.array([interp_x(handles[0], y) for y in ys])
    x1 = np.array([interp_x(handles[1], y) for y in ys])
    return float(np.mean(np.abs(x0 - x1))) / pixels_per_mm


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python edge_detection.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    try:
        pixels_per_mm = float(np.load(HOMOGRAPHY_PATH)["pixels_per_mm"])
        print(f"Loaded pixels_per_mm={pixels_per_mm} from {HOMOGRAPHY_PATH}")
    except Exception:
        pixels_per_mm = None
        print(f"Warning: could not load {HOMOGRAPHY_PATH} — diameter will be shown in pixels")

    L_ch, a_ch, b_ch = to_lab_channels(img)
    h, w = img.shape[:2]

    WIN_GRAD     = "Combined gradient"
    WIN_FILTERED = "Filtered gradient"
    WIN_LINES    = "Two best lines"
    WIN_IDEAL    = "Idealised lines"

    for win in (WIN_GRAD, WIN_FILTERED, WIN_LINES, WIN_IDEAL):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    # Shared mutable state for drag interaction
    state = {"handles": None, "dragging": None}

    def redraw(print_result=False):
        ppm  = pixels_per_mm if pixels_per_mm else 1.0
        unit = "mm" if pixels_per_mm else "px"
        ideal = draw_handles(img, state["handles"])
        diam = compute_diameter_from_pts(state["handles"], ppm)
        if diam is not None:
            label = f"Diameter: {diam:.2f} {unit}"
            if print_result:
                print(label)
            cv2.putText(ideal, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
            cv2.putText(ideal, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow(WIN_IDEAL, ideal)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["handles"] is None:
                return
            best_dist, best_key = float("inf"), None
            for li, pts in enumerate(state["handles"]):
                if pts is None:
                    continue
                for pi, pt in enumerate(pts):
                    d = ((pt[0] - x) ** 2 + (pt[1] - y) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best_key = d, (li, pi)
            if best_dist <= DRAG_RADIUS * 2:
                state["dragging"] = best_key
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            li, pi = state["dragging"]
            state["handles"][li][pi] = [x, y]
            redraw(print_result=False)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"] is not None:
            state["dragging"] = None
            redraw(print_result=True)

    cv2.setMouseCallback(WIN_IDEAL, on_mouse)

    def update(_=None):
        w_L           = cv2.getTrackbarPos("L weight",     WIN_GRAD) / 100.0
        w_a           = cv2.getTrackbarPos("a weight",     WIN_GRAD) / 100.0
        w_b           = cv2.getTrackbarPos("b weight",     WIN_GRAD) / 100.0
        blur_sigma    = cv2.getTrackbarPos("Blur sigma",   WIN_GRAD)
        ksize         = [1, 3, 5, 7][cv2.getTrackbarPos("Sobel ksize", WIN_GRAD)]
        threshold_pct = cv2.getTrackbarPos("Threshold %",   WIN_FILTERED)
        min_line_pct  = cv2.getTrackbarPos("Min line span", WIN_FILTERED)

        # 1. Preprocessing
        blurred = blur_channels((L_ch, a_ch, b_ch), blur_sigma)

        # 2. Gradient calculation
        grads      = channel_gradients(blurred, ksize)
        norm_grads = normalise_gradients(grads)
        combined   = combine_gradients(norm_grads, (w_L, w_a, w_b))

        grad_display = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow(WIN_GRAD, grad_display)

        # 3. Line detection
        mask = threshold_gradient(combined, threshold_pct)

        filtered = np.zeros_like(grad_display)
        filtered[mask] = grad_display[mask]
        cv2.imshow(WIN_FILTERED, filtered)

        min_votes = max(1, int(h * min_line_pct / 100))
        line1, line2 = find_best_lines(mask, min_votes)

        # 4. Visualisation
        cv2.imshow(WIN_LINES, make_overlay(img, mask, line1, line2))

        # Convert Hough lines to endpoint handles and reset drag state
        def line_to_handle(line):
            if line is None:
                return None
            pt_top, pt_bot = line_endpoints(line, h, w)
            return [list(pt_top), list(pt_bot)]

        state["handles"] = [line_to_handle(line1), line_to_handle(line2)]
        state["dragging"] = None
        redraw(print_result=True)

    cv2.createTrackbar("L weight",      WIN_GRAD,     100, 100, update)
    cv2.createTrackbar("a weight",      WIN_GRAD,      50, 100, update)
    cv2.createTrackbar("b weight",      WIN_GRAD,      50, 100, update)
    cv2.createTrackbar("Blur sigma",    WIN_GRAD,       2,  20, update)
    cv2.createTrackbar("Sobel ksize",   WIN_GRAD,       1,   3, update)
    cv2.createTrackbar("Threshold %",   WIN_FILTERED,  10, 100, update)
    cv2.createTrackbar("Min line span", WIN_FILTERED,  20, 100, update)
    update()

    print("Adjust sliders to auto-detect lines. Drag handles in 'Idealised lines' to fine-tune.")
    print("Press any key to quit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
