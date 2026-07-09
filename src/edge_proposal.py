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


def _normalize_channels(grads):
    return tuple(g / (np.percentile(g, _NORMALIZATION_PERCENTILE) or 1.0) for g in grads)


def _combine(grads, weights):
    return sum(w * g for w, g in zip(weights, grads))


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


def _hough_candidates(mask, min_votes):
    """Near-vertical Hough line candidates, strongest (most-voted) first — cv2.HoughLines
    already returns lines in decreasing order of accumulator votes.
    """
    lines = cv2.HoughLines(mask.astype(np.uint8) * 255,
                           rho=1, theta=np.pi / 900, threshold=min_votes)
    if lines is None:
        return []
    return [_normalize_near_vertical(float(r), float(t)) for r, t in lines[:, 0]
            if t < np.pi / 12 or t > 11 * np.pi / 12]


def _select_pair(candidates, min_sep_px, h):
    if not candidates:
        return None, None
    line1 = candidates[0]
    line2 = next((l for l in candidates[1:] if _well_separated(line1, l, h, min_sep_px)), None)
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


def _draw_candidates(mask, candidates, h, w):
    out = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for line in candidates:
        pt_top, pt_bot = _endpoints(line, h, w)
        cv2.line(out, pt_top, pt_bot, (0, 255, 255), 1)
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
    combined = _combine(_normalize_channels(grads), weights)
    thinned = _suppress_non_maxima(combined)
    mask = _threshold(thinned, threshold_pct)

    min_sep_px = min_sep_mm * pixels_per_mm
    min_votes = max(1, int(h * min_line_span_pct / 100))
    candidates = _hough_candidates(mask, min_votes)
    line1, line2 = _select_pair(candidates, min_sep_px, h)
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


def _show(name, image, w, h):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, max(1, int(800 * h / w)))
    cv2.imshow(name, image)


def _to_display(values):
    """Scale a continuous gradient map to 0-255 using the same percentile white point the
    algorithm itself uses (see _threshold), instead of the true min/max — a true min/max
    stretch lets one rare outlier pixel crush every real edge down near black.
    """
    scale = float(np.percentile(values, _NORMALIZATION_PERCENTILE)) or 1.0
    return np.clip(values / scale * 255, 0, 255).astype(np.uint8)


def propose_test(image, pixels_per_mm, weights=_DEFAULT_WEIGHTS, blur_sigma=_DEFAULT_BLUR_SIGMA,
                  sobel_ksize=_DEFAULT_SOBEL_KSIZE, threshold_pct=_DEFAULT_THRESHOLD_PCT,
                  min_line_span_pct=_DEFAULT_MIN_LINE_SPAN_PCT, min_sep_mm=_MIN_SEP_MM):
    """Non-interactive edge detection that displays every pipeline stage for inspection.

    Same algorithm and parameters as `propose_headless` — no sliders, you pass in exactly
    the values you want to test (e.g. from named constants in orchestrate.py) — but opens
    one window per stage (original, blurred, unnormalised gradient, normalised gradient,
    non-max suppressed, threshold mask with the top 10 Hough line candidates overlaid,
    final selected edges on the original image) instead of running silently, so you can
    see the effect of a specific parameter set. Press any key to close the windows and
    continue.

    Returns [[top0, bot0], [top1, bot1]], or None if fewer than two edges were found.
    """
    L_ch, a_ch, b_ch = _to_lab(image)
    h, w = image.shape[:2]

    blurred = _blur((L_ch, a_ch, b_ch), blur_sigma)
    grads = _gradients(blurred, sobel_ksize)
    raw_combined = _combine(grads, weights)
    normalised = _combine(_normalize_channels(grads), weights)
    thinned = _suppress_non_maxima(normalised)
    mask = _threshold(thinned, threshold_pct)

    min_sep_px = min_sep_mm * pixels_per_mm
    min_votes = max(1, int(h * min_line_span_pct / 100))
    candidates = _hough_candidates(mask, min_votes)
    line1, line2 = _select_pair(candidates, min_sep_px, h)

    _show("1. Original image", image, w, h)
    _show("2. Blurred image", cv2.cvtColor(cv2.merge(blurred), cv2.COLOR_LAB2BGR), w, h)
    _show("3. Unnormalised gradient", _to_display(raw_combined), w, h)
    _show("4. Normalised gradient", _to_display(normalised), w, h)
    _show("5. Non-maximum suppressed", _to_display(thinned), w, h)
    _show("6. Threshold mask + top 10 candidates", _draw_candidates(mask, candidates[:50], h, w), w, h)
    _show("7. Final edges", _draw_lines(image, line1, line2, h, w), w, h)

    print("Showing pipeline stages — press any key in a window to close and continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if line1 is None or line2 is None:
        return None
    return [_to_handle(line1, h, w), _to_handle(line2, h, w)]
