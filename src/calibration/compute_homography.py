import cv2
import numpy as np
import os

CAMERA_INDEX = 4
CAMERA_RESOLUTION = (3840, 2160)  # must match the resolution used during calibration
DISPLAY_SIZE = (3840, 2160)       # must stay equal to CAMERA_RESOLUTION
CALIBRATION_PATH = "data/calibration.npz"
OUTPUT_PATH = "data/homography.npz"

# ── Measure your printed AprilTag carefully with a ruler ─────────────────────
TAG_SIZE_MM = 46.0   # side length of the black square of your printed tag (not the white border)
APRILTAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
PIXELS_PER_MM = 10.0
PADDING_MM = 150.0   # extra scene shown around the tag on every side
# ─────────────────────────────────────────────────────────────────────────────

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]


def draw_detection(img, corners):
    display = img.copy()
    pts = corners.astype(int)
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    for label, pt in zip(CORNER_LABELS, pts):
        cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
        cv2.putText(display, label, (pt[0] + 12, pt[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    msg = "Tag detected — press ENTER to confirm, any other key to retry"
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


def detect_tag_corners(frame):
    """Detect a single AprilTag in the frame.

    Returns its 4 corners as (top-left, top-right, bottom-right, bottom-left),
    or None if zero tags were found. If multiple tags are visible, uses the first one.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILY)
    detector_params = cv2.aruco.DetectorParameters()
    # Corners are used to solve an exact 4-point homography, so any detection error goes
    # straight into it unaveraged — sub-pixel refinement matters a lot more here than in
    # typical ArUco use (e.g. pose estimation), where it's usually left off.
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) == 0:
        print("No AprilTag detected — make sure it's flat, well-lit, and fully visible.")
        return None
    if len(ids) > 1:
        print(f"Multiple tags detected ({len(ids)}); using the first one (id={ids[0][0]}).")
    return corners[0][0]  # shape (4, 2): tl, tr, br, bl


def main():
    data = np.load(CALIBRATION_PATH)
    K, dist = data["K"], data["dist"]

    print("Place your AprilTag flat in the scene, at the exact plane you want to calibrate.")
    print("Press SPACE to freeze the frame when ready.")

    points = None
    while points is None:
        frame = capture_frame(K, dist)
        if frame is None:
            return

        tag_corners = detect_tag_corners(frame)
        if tag_corners is None:
            continue

        cv2.namedWindow("Detected tag", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Detected tag", draw_detection(frame, tag_corners))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == 13:  # ENTER
            points = tag_corners

    ref_w = round(TAG_SIZE_MM * PIXELS_PER_MM)
    ref_h = round(TAG_SIZE_MM * PIXELS_PER_MM)
    pad   = round(PADDING_MM  * PIXELS_PER_MM)

    # Place the tag at (pad, pad) inside a larger canvas.
    # Everything within PADDING_MM of the tag is also visible.
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

    # --- Camera pose relative to the tag/measurement frame -----------------------------
    # Independent of H above: same 4 corners, but solved as a full 3D pose (via the tag's
    # known physical size) rather than a flat pixel-to-pixel warp. solvePnP returns the
    # tag -> camera transform; invert it to get the camera's position/orientation expressed
    # in the tag's own coordinate frame (origin at its top-left corner, X along its top
    # edge, Y along its left edge).
    tag_mm_3d = np.array([
        [0, 0, 0],
        [TAG_SIZE_MM, 0, 0],
        [TAG_SIZE_MM, TAG_SIZE_MM, 0],
        [0, TAG_SIZE_MM, 0],
    ], dtype=np.float32)
    # src's corners come from `frame`, which capture_frame() already ran through
    # cv2.undistort(f, K, dist) - so no distortion remains to correct for here. Passing the
    # real `dist` again would tell solvePnP to undistort already-undistorted points a second
    # time, skewing the recovered pose.
    ok, rvec, tvec = cv2.solvePnP(tag_mm_3d, src, K, None)
    if not ok:
        print("solvePnP failed - skipping pose computation.")
        camera_position_mm = None
        camera_rotation_deg = None
    else:
        R_tag_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_in_tag = R_tag_to_cam.T
        camera_position_mm = -R_cam_in_tag @ tvec.flatten()
        # Euler angles (degrees) such that R_cam_in_tag = Rz(z) @ Ry(y) @ Rx(x) - verified
        # against cv2.RQDecomp3x3's actual convention, not assumed.
        camera_rotation_deg, *_ = cv2.RQDecomp3x3(R_cam_in_tag)
        print(f"\nCamera position in tag frame (mm):   x={camera_position_mm[0]:.1f}  "
              f"y={camera_position_mm[1]:.1f}  z={camera_position_mm[2]:.1f}")
        print(f"Camera rotation in tag frame (deg):  x={camera_rotation_deg[0]:.2f}  "
              f"y={camera_rotation_deg[1]:.2f}  z={camera_rotation_deg[2]:.2f}")
        if camera_position_mm[2] > 0:
            print("WARNING: camera_position_mm z is positive - expected negative given the "
                  "tag frame's Z axis points away from the camera (see convention notes). "
                  "Double-check the tag was detected right-side-up before trusting this.")

    flat = cv2.warpPerspective(frame, H, (full_w, full_h))
    cv2.namedWindow("BEV preview - press any key to save", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV preview - press any key to save", 800, round(800 * full_h / full_w))
    cv2.imshow("BEV preview - press any key to save", flat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if camera_position_mm is None:
        print("Not saving - pose computation failed (see warning above).")
        return

    os.makedirs("data", exist_ok=True)
    # H itself is no longer saved - it's reconstructed at runtime from camera_position_mm/
    # camera_rotation_deg (plus K/dist and this canvas layout), so the app can let camera
    # position/orientation be adjusted directly instead of raw, unintuitive matrix entries.
    np.savez(
        OUTPUT_PATH,
        camera_position_mm=camera_position_mm,
        camera_rotation_deg=np.array(camera_rotation_deg),
        output_w=np.int32(full_w),
        output_h=np.int32(full_h),
        pixels_per_mm=np.float64(PIXELS_PER_MM),
        canvas_origin_px=np.array([pad, pad], dtype=np.float64),
    )
    print(f"\nSaved to {OUTPUT_PATH}  ({full_w}x{full_h}px at {PIXELS_PER_MM}px/mm)")


if __name__ == "__main__":
    main()
