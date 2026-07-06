import cv2

CAMERA_INDICES = [0, 1, 2, 3, 4, 5]

RESOLUTIONS = [
    (640,  480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

FOURCC_OPTIONS = [
    ("default", None),
    ("MJPG",    cv2.VideoWriter_fourcc(*"MJPG")),
    ("YUYV",    cv2.VideoWriter_fourcc(*"YUYV")),
]


def probe_once(index, fourcc_val, width, height):
    """Open camera fresh, request settings, read one frame, close."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    if fourcc_val is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Drain a few frames so the driver applies the new settings
    for _ in range(3):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    rh, rw = frame.shape[:2]
    return rw, rh


def main():
    supported = []  # (index, fourcc_name, fourcc_val, w, h)

    for idx in CAMERA_INDICES:
        # Quick check: does this index exist?
        cap = cv2.VideoCapture(idx)
        exists = cap.isOpened()
        cap.release()
        if not exists:
            continue

        print(f"\n── Camera {idx} ──")
        for fourcc_name, fourcc_val in FOURCC_OPTIONS:
            for w, h in RESOLUTIONS:
                result = probe_once(idx, fourcc_val, w, h)
                if result is None:
                    status = "no frame"
                else:
                    rw, rh = result
                    if (rw, rh) == (w, h):
                        status = f"OK  → {rw}x{rh}"
                        supported.append((idx, fourcc_name, fourcc_val, w, h))
                    else:
                        status = f"fallback → {rw}x{rh}"
                print(f"  [{fourcc_name:7s}] {w:4d}x{h:<4d}  {status}")

    print("\n" + "═" * 50)
    if not supported:
        print("No exact resolution match found on any camera.")
        print("Opening camera 1 at its default resolution for preview.")
        preview_args = (1, "default", None, None, None)
    else:
        print("Supported combinations (index, fourcc, resolution):")
        for i, (idx, fn, _, w, h) in enumerate(supported):
            print(f"  [{i}]  camera {idx}  {fn:7s}  {w}x{h}")
        preview_args = max(supported, key=lambda x: x[3] * x[4])  # highest pixel count

    # ── Live preview ──────────────────────────────────────────────────────────
    idx, fourcc_name, fourcc_val, w, h = preview_args
    cap = cv2.VideoCapture(idx)
    if fourcc_val is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
    if w is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(3):
        cap.read()

    label = f"{w}x{h} {fourcc_name}" if w else "default"
    print(f"\nShowing camera {idx} at {label} — press q to quit.")

    cv2.namedWindow("Camera test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera test", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        fh, fw = frame.shape[:2]
        cv2.putText(frame, f"cam={idx}  {fw}x{fh}  {fourcc_name}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Camera test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
