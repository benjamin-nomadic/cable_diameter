import copy
import cv2

DRAG_RADIUS = 20


def _draw(bgr, handles):
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


def refine(image, handles, compute_diameter=None):
    """Show draggable endpoint handles for manual edge adjustment.

    compute_diameter: optional callable(handles) -> float mm, shown as live readout.
    Press SPACE or ENTER to confirm.  Press Q or ESC to skip (returns original handles).
    Returns [[top0, bot0], [top1, bot1]].
    """
    state = {"handles": copy.deepcopy(handles), "dragging": None}

    WIN = "Refinement"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    def redraw():
        img = _draw(image, state["handles"])
        if compute_diameter is not None:
            diam = compute_diameter(state["handles"])
            if diam is not None:
                label = f"Diameter: {diam:.2f} mm"
                cv2.putText(img, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
                cv2.putText(img, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow(WIN, img)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            best_d, best_key = float("inf"), None
            for li, pts in enumerate(state["handles"]):
                if pts is None:
                    continue
                for pi, pt in enumerate(pts):
                    d = ((pt[0] - x) ** 2 + (pt[1] - y) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_key = d, (li, pi)
            if best_d <= DRAG_RADIUS * 2:
                state["dragging"] = best_key
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            li, pi = state["dragging"]
            state["handles"][li][pi] = [x, y]
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"] is not None:
            state["dragging"] = None
            redraw()

    cv2.setMouseCallback(WIN, on_mouse)
    redraw()

    print("Drag handles to refine edges.  SPACE/ENTER = confirm  Q/ESC = skip refinement.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord(" "), 13):
            cv2.destroyWindow(WIN)
            return state["handles"]
        elif key in (ord("q"), 27):
            cv2.destroyWindow(WIN)
            return handles
