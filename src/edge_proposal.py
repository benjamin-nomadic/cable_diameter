import cv2
import numpy as np

_MIN_SEP_MM = 5   # minimum physical gap between the two accepted lines, checked at both ends
_NORMALIZATION_PERCENTILE = 99   # scale by this percentile instead of the true max, so a
                                  # single hot/saturated pixel can't skew the whole frame's scale


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_lab(bgr):
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))


def _blur(channels, sigma):
    if sigma <= 0:
        return channels
    return tuple(cv2.GaussianBlur(ch, (0, 0), sigma) for ch in channels)


def _gradients(channels, ksize):
    return tuple(np.abs(cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=ksize)) for ch in channels)


def _combine(grads, weights):
    norm = tuple(g / (np.percentile(g, _NORMALIZATION_PERCENTILE) or 1.0) for g in grads)
    return sum(w * g for w, g in zip(weights, norm))


def _threshold(combined, pct):
    scale = float(np.percentile(combined, _NORMALIZATION_PERCENTILE)) or 1.0
    return combined >= (pct / 100.0) * scale


def _suppress_non_maxima(combined):
    """Thin the gradient map to single-pixel-wide ridges along each row.

    Blur spreads a real edge's gradient across several pixels, so thresholding the raw
    map produces a several-pixels-wide band rather than a precise line — this keeps a
    pixel only if it's >= both its immediate left and right neighbor in the same row
    (the direction we expect near-vertical edges to vary across), zeroing everything
    else. Edge-of-row pixels are compared against a padding of 0 on the missing side.
    """
    left = np.zeros_like(combined)
    left[:, 1:] = combined[:, :-1]
    right = np.zeros_like(combined)
    right[:, :-1] = combined[:, 1:]
    is_peak = (combined >= left) & (combined >= right)
    return np.where(is_peak, combined, 0)


def _normalize_near_vertical(rho, theta):
    """Collapse the theta≈0 and theta≈π representations of the same near-vertical line
    into one consistent (rho, theta) form, so the same physical line always compares equal.
    """
    if theta > np.pi / 2:
        return -rho, theta - np.pi
    return rho, theta


def _x_at_y(line, y):
    """Unclipped x-position of a (rho, theta) line at height y — used only to check
    separation, as opposed to `_endpoints`, which clips to the image bounds for display.
    """
    rho, theta = line
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if abs(cos_t) < 1e-6:
        return rho
    return (rho - y * sin_t) / cos_t


def _well_separated(line_a, line_b, h, min_sep_px):
    """True if line_b stays at least min_sep_px from line_a at BOTH the top and bottom
    of the image, on the same side throughout — i.e. they don't cross or converge
    anywhere in between. Checking only one reference point (e.g. rho) can't catch two
    lines that are far apart at the top but cross before the bottom.
    """
    top_gap = _x_at_y(line_b, 0) - _x_at_y(line_a, 0)
    bot_gap = _x_at_y(line_b, h - 1) - _x_at_y(line_a, h - 1)
    return top_gap * bot_gap > 0 and abs(top_gap) >= min_sep_px and abs(bot_gap) >= min_sep_px


def _hough(mask, min_votes, min_sep_px, h):
    lines = cv2.HoughLines(mask.astype(np.uint8) * 255,
                           rho=1, theta=np.pi / 900, threshold=min_votes)
    if lines is None:
        return None, None
    near_vert = [_normalize_near_vertical(float(r), float(t)) for r, t in lines[:, 0]
                 if t < np.pi / 12 or t > 11 * np.pi / 12]
    if not near_vert:
        return None, None
    line1 = near_vert[0]
    line2 = next((l for l in near_vert[1:] if _well_separated(line1, l, h, min_sep_px)), None)
    return line1, line2


def _endpoints(line, h, w):
    rho, theta = line
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if abs(cos_t) < 1e-6:
        return (int(rho), 0), (int(rho), h - 1)
    x_top = int(np.clip(rho / cos_t, 0, w - 1))
    x_bot = int(np.clip((rho - (h - 1) * sin_t) / cos_t, 0, w - 1))
    return (x_top, 0), (x_bot, h - 1)


def _to_handle(line, h, w):
    if line is None:
        return None
    pt_top, pt_bot = _endpoints(line, h, w)
    return [list(pt_top), list(pt_bot)]


def _draw_lines(bgr, line1, line2, h, w):
    out = bgr.copy()
    for line, colour in [(line1, (0, 220, 0)), (line2, (0, 60, 255))]:
        if line is None:
            continue
        pt_top, pt_bot = _endpoints(line, h, w)
        cv2.line(out, pt_top, pt_bot, colour, 2)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

# Defaults mirror the interactive sliders' starting positions in `propose`.
_DEFAULT_WEIGHTS = (1.0, 0.5, 0.5)
_DEFAULT_BLUR_SIGMA = 7
_DEFAULT_SOBEL_KSIZE = 3
_DEFAULT_THRESHOLD_PCT = 50
_DEFAULT_MIN_LINE_SPAN_PCT = 20


def propose_headless(image, pixels_per_mm, weights=_DEFAULT_WEIGHTS, blur_sigma=_DEFAULT_BLUR_SIGMA,
                      sobel_ksize=_DEFAULT_SOBEL_KSIZE, threshold_pct=_DEFAULT_THRESHOLD_PCT,
                      min_line_span_pct=_DEFAULT_MIN_LINE_SPAN_PCT, min_sep_mm=_MIN_SEP_MM):
    """Non-interactive edge detection — same algorithm as `propose`, fixed parameters.

    pixels_per_mm: calibration scale factor, used to enforce min_sep_mm physically
    rather than as a raw pixel count.
    Returns [[top0, bot0], [top1, bot1]], or None if fewer than two edges were found.
    """
    L_ch, a_ch, b_ch = _to_lab(image)
    h, w = image.shape[:2]

    blurred = _blur((L_ch, a_ch, b_ch), blur_sigma)
    grads = _gradients(blurred, sobel_ksize)
    combined = _combine(grads, weights)
    thinned = _suppress_non_maxima(combined)
    mask = _threshold(thinned, threshold_pct)

    min_sep_px = min_sep_mm * pixels_per_mm
    line1, line2 = _hough(mask, max(1, int(h * min_line_span_pct / 100)), min_sep_px, h)
    if line1 is None or line2 is None:
        return None
    return [_to_handle(line1, h, w), _to_handle(line2, h, w)]


_DEFAULT_HANDLE_X_FRACTIONS = (0.4, 0.6)


def default_handles(image, x_fractions=_DEFAULT_HANDLE_X_FRACTIONS):
    """Fallback edge positions for when automatic detection finds nothing sensible.

    Places two vertical lines at fixed fractions of the image width, spanning its full
    height, so there's always something reasonable to drag into place by hand instead
    of failing outright. Same [[top0, bot0], [top1, bot1]] shape as `propose_headless`.
    """
    h, w = image.shape[:2]
    x_left, x_right = (int(w * f) for f in x_fractions)
    return [[[x_left, 0], [x_left, h - 1]], [[x_right, 0], [x_right, h - 1]]]


def propose(image, pixels_per_mm, min_sep_mm=_MIN_SEP_MM):
    """Interactively detect two cable edges via LAB gradient + Hough lines.

    Shows three windows with adjustment sliders.
    Press SPACE or ENTER to accept the current proposal.
    Press Q or ESC to cancel.

    pixels_per_mm: calibration scale factor, used to enforce min_sep_mm physically
    rather than as a raw pixel count.
    Returns [[top0, bot0], [top1, bot1]] on accept, None on cancel.
    Each point is a [x, y] list.
    """
    min_sep_px = min_sep_mm * pixels_per_mm
    L_ch, a_ch, b_ch = _to_lab(image)
    h, w = image.shape[:2]

    WIN_GRAD = "Proposal: gradient"
    WIN_FILT = "Proposal: filtered"
    WIN_PROP = "Proposal: detected edges"

    for win in (WIN_GRAD, WIN_FILT, WIN_PROP):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    state = {"handles": None}

    def update(_=None):
        wL  = cv2.getTrackbarPos("L weight",      WIN_GRAD) / 100.0
        wa  = cv2.getTrackbarPos("a weight",      WIN_GRAD) / 100.0
        wb  = cv2.getTrackbarPos("b weight",      WIN_GRAD) / 100.0
        sig = cv2.getTrackbarPos("Blur sigma",    WIN_GRAD)
        ks  = [1, 3, 5, 7][cv2.getTrackbarPos("Sobel ksize", WIN_GRAD)]
        thr = cv2.getTrackbarPos("Threshold %",   WIN_FILT)
        mlp = cv2.getTrackbarPos("Min line span", WIN_FILT)

        blurred  = _blur((L_ch, a_ch, b_ch), sig)
        grads    = _gradients(blurred, ks)
        combined = _combine(grads, (wL, wa, wb))

        grad_u8 = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow(WIN_GRAD, grad_u8)

        thinned = _suppress_non_maxima(combined)
        mask = _threshold(thinned, thr)
        filtered = np.zeros_like(grad_u8)
        filtered[mask] = grad_u8[mask]
        cv2.imshow(WIN_FILT, filtered)

        line1, line2 = _hough(mask, max(1, int(h * mlp / 100)), min_sep_px, h)
        cv2.imshow(WIN_PROP, _draw_lines(image, line1, line2, h, w))

        state["handles"] = [_to_handle(line1, h, w), _to_handle(line2, h, w)]

    cv2.createTrackbar("L weight",      WIN_GRAD,   100, 100, update)
    cv2.createTrackbar("a weight",      WIN_GRAD,    50, 100, update)
    cv2.createTrackbar("b weight",      WIN_GRAD,    50, 100, update)
    cv2.createTrackbar("Blur sigma",    WIN_GRAD,     2,  20, update)
    cv2.createTrackbar("Sobel ksize",   WIN_GRAD,     1,   3, update)
    cv2.createTrackbar("Threshold %",   WIN_FILT,    10, 100, update)
    cv2.createTrackbar("Min line span", WIN_FILT,    20, 100, update)
    update()

    print("Adjust sliders to detect edges.  SPACE/ENTER = accept  Q/ESC = cancel.")
    result = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord(" "), 13):
            result = state["handles"]
            print("Accepted with these parameters — copy into edge_proposal.py's _DEFAULT_* constants:")
            print(f"  _DEFAULT_WEIGHTS = ({cv2.getTrackbarPos('L weight', WIN_GRAD) / 100.0}, "
                  f"{cv2.getTrackbarPos('a weight', WIN_GRAD) / 100.0}, "
                  f"{cv2.getTrackbarPos('b weight', WIN_GRAD) / 100.0})")
            print(f"  _DEFAULT_BLUR_SIGMA = {cv2.getTrackbarPos('Blur sigma', WIN_GRAD)}")
            print(f"  _DEFAULT_SOBEL_KSIZE = {[1, 3, 5, 7][cv2.getTrackbarPos('Sobel ksize', WIN_GRAD)]}")
            print(f"  _DEFAULT_THRESHOLD_PCT = {cv2.getTrackbarPos('Threshold %', WIN_FILT)}")
            print(f"  _DEFAULT_MIN_LINE_SPAN_PCT = {cv2.getTrackbarPos('Min line span', WIN_FILT)}")
            break
        elif key in (ord("q"), 27):
            break

    for win in (WIN_GRAD, WIN_FILT, WIN_PROP):
        cv2.destroyWindow(win)
    return result
