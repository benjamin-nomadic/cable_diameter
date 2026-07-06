import cv2
import numpy as np

_MIN_SEP = 40   # minimum rho gap (px) between the two accepted Hough lines


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_lab(bgr):
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))


def _blur(channels, sigma):
    if sigma == 0:
        return channels
    return tuple(cv2.GaussianBlur(ch, (0, 0), sigma) for ch in channels)


def _gradients(channels, ksize):
    return tuple(np.abs(cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=ksize)) for ch in channels)


def _combine(grads, weights):
    norm = tuple(g / (g.max() or 1.0) for g in grads)
    return sum(w * g for w, g in zip(weights, norm))


def _threshold(combined, pct):
    return combined >= (pct / 100.0) * (float(combined.max()) or 1.0)


def _hough(mask, min_votes):
    lines = cv2.HoughLines(mask.astype(np.uint8) * 255,
                           rho=1, theta=np.pi / 900, threshold=min_votes)
    if lines is None:
        return None, None
    near_vert = [(float(r), float(t)) for r, t in lines[:, 0]
                 if t < np.pi / 12 or t > 11 * np.pi / 12]
    if not near_vert:
        return None, None
    line1 = near_vert[0]
    line2 = next((l for l in near_vert[1:] if abs(l[0] - line1[0]) > _MIN_SEP), None)
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
_DEFAULT_BLUR_SIGMA = 2
_DEFAULT_SOBEL_KSIZE = 3
_DEFAULT_THRESHOLD_PCT = 10
_DEFAULT_MIN_LINE_SPAN_PCT = 20


def propose_headless(image, weights=_DEFAULT_WEIGHTS, blur_sigma=_DEFAULT_BLUR_SIGMA,
                      sobel_ksize=_DEFAULT_SOBEL_KSIZE, threshold_pct=_DEFAULT_THRESHOLD_PCT,
                      min_line_span_pct=_DEFAULT_MIN_LINE_SPAN_PCT):
    """Non-interactive edge detection — same algorithm as `propose`, fixed parameters.

    Returns [[top0, bot0], [top1, bot1]], or None if fewer than two edges were found.
    """
    L_ch, a_ch, b_ch = _to_lab(image)
    h, w = image.shape[:2]

    blurred = _blur((L_ch, a_ch, b_ch), blur_sigma)
    grads = _gradients(blurred, sobel_ksize)
    combined = _combine(grads, weights)
    mask = _threshold(combined, threshold_pct)

    line1, line2 = _hough(mask, max(1, int(h * min_line_span_pct / 100)))
    if line1 is None or line2 is None:
        return None
    return [_to_handle(line1, h, w), _to_handle(line2, h, w)]


def propose(image):
    """Interactively detect two cable edges via LAB gradient + Hough lines.

    Shows three windows with adjustment sliders.
    Press SPACE or ENTER to accept the current proposal.
    Press Q or ESC to cancel.

    Returns [[top0, bot0], [top1, bot1]] on accept, None on cancel.
    Each point is a [x, y] list.
    """
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

        mask = _threshold(combined, thr)
        filtered = np.zeros_like(grad_u8)
        filtered[mask] = grad_u8[mask]
        cv2.imshow(WIN_FILT, filtered)

        line1, line2 = _hough(mask, max(1, int(h * mlp / 100)))
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
            break
        elif key in (ord("q"), 27):
            break

    for win in (WIN_GRAD, WIN_FILT, WIN_PROP):
        cv2.destroyWindow(win)
    return result
