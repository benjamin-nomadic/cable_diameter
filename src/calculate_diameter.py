import numpy as np

N_SAMPLES = 500


def calculate(handles, pixels_per_mm, height_mm=None):
    """Compute the true cable diameter from two edge line segments.

    handles:      [[top0, bot0], [top1, bot1]] where each point is [x, y].
    pixels_per_mm: calibration scale factor.
    height_mm:    if provided, applies the geometric correction
                      d_true = 2*(m²/h) + sqrt(m⁴/h² + m²)
                  where m is the raw measured horizontal distance and h is the
                  known physical height of the camera above the cable plane.
                  If None, returns m directly (uncorrected).

    Returns diameter in mm, or None if the segments don't overlap vertically.
    """
    if handles is None or handles[0] is None or handles[1] is None:
        return None

    def interp_x(pts, y):
        (xt, yt), (xb, yb) = pts
        if yb == yt:
            return float(xt)
        return xt + (xb - xt) * (y - yt) / (yb - yt)

    y0 = sorted([handles[0][0][1], handles[0][1][1]])
    y1 = sorted([handles[1][0][1], handles[1][1][1]])
    y_min = max(y0[0], y1[0])
    y_max = min(y0[1], y1[1])
    if y_min >= y_max:
        return None

    ys = np.linspace(y_min, y_max, N_SAMPLES)
    x0 = np.array([interp_x(handles[0], y) for y in ys])
    x1 = np.array([interp_x(handles[1], y) for y in ys])
    m = float(np.mean(np.abs(x0 - x1))) / pixels_per_mm

    if height_mm is None:
        return m

    h = height_mm
    return 2.0 * (m ** 2 / h) + np.sqrt(m ** 4 / h ** 2 + m ** 2)
