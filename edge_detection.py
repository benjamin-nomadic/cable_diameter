import sys
import cv2
import numpy as np

MIN_SEPARATION = 40   # minimum rho gap (px) between the two selected lines
BIN_HALF_WIDTH = 5    # pixels either side of a line to highlight in the overlay


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

def draw_line(img, rho, theta, colour, thickness=2):
    """Draw a full-image Hough line (rho, theta) onto img."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x0, y0 = cos_t * rho, sin_t * rho
    scale = max(img.shape)
    pt1 = (int(x0 + scale * (-sin_t)), int(y0 + scale * cos_t))
    pt2 = (int(x0 - scale * (-sin_t)), int(y0 - scale * cos_t))
    cv2.line(img, pt1, pt2, colour, thickness)


def make_overlay(bgr, mask, line1, line2):
    """Colour the mask pixels near each detected line and draw the line itself."""
    overlay = bgr.copy()
    ys, xs = np.where(mask)
    for line, colour in [(line1, (0, 220, 0)), (line2, (0, 60, 255))]:
        if line is None:
            continue
        rho, theta = line
        dists = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
        overlay[ys[dists <= BIN_HALF_WIDTH], xs[dists <= BIN_HALF_WIDTH]] = colour
        draw_line(overlay, rho, theta, colour)
    return overlay


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python edge_detection.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    # Preprocessing step that only needs to run once
    L_ch, a_ch, b_ch = to_lab_channels(img)

    WIN_GRAD     = "Combined gradient"
    WIN_FILTERED = "Filtered gradient"
    WIN_LINES    = "Two best lines"

    for win in (WIN_GRAD, WIN_FILTERED, WIN_LINES):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    def update(_=None):
        # Read all slider values
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

        min_votes = max(1, int(L_ch.shape[0] * min_line_pct / 100))
        line1, line2 = find_best_lines(mask, min_votes)

        # 4. Visualisation
        cv2.imshow(WIN_LINES, make_overlay(img, mask, line1, line2))

    cv2.createTrackbar("L weight",    WIN_GRAD,     100, 100, update)
    cv2.createTrackbar("a weight",    WIN_GRAD,      50, 100, update)
    cv2.createTrackbar("b weight",    WIN_GRAD,      50, 100, update)
    cv2.createTrackbar("Blur sigma",  WIN_GRAD,       2,  20, update)
    cv2.createTrackbar("Sobel ksize", WIN_GRAD,       1,   3, update)  # 0=1,1=3,2=5,3=7
    cv2.createTrackbar("Threshold %",   WIN_FILTERED, 10, 100, update)
    cv2.createTrackbar("Min line span", WIN_FILTERED, 20, 100, update)
    update()

    print("Adjust sliders. Press any key to quit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
