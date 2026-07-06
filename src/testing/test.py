import cv2
import numpy as np
import os

CAMERA_INDEX = 4
RESOLUTION = (3840, 2160)  # must match the resolution used during calibration
CALIBRATION_PATH = "data/calibration.npz"
HOMOGRAPHY_PATH = "data/homography.npz"
PANEL_SIZE = (1280, 720)   # display size per panel (3 panels side by side)


def load_calibration():
    data = np.load(CALIBRATION_PATH)
    return data["K"], data["dist"]


def load_homography():
    if not os.path.exists(HOMOGRAPHY_PATH):
        return None, None
    data = np.load(HOMOGRAPHY_PATH)
    return data["H"], (int(data["output_w"]), int(data["output_h"]))


def main():
    K, dist = load_calibration()
    H, bev_size = load_homography()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    if not cap.isOpened():
        print(f"Could not open camera at index {CAMERA_INDEX}")
        return

    if H is not None:
        print("Homography loaded — showing Raw | Undistorted | BEV")
        window = "Raw | Undistorted | BEV"
        win_w, win_h = 1920, 360
    else:
        print("No homography found — run src/calibration/compute_homography.py to add BEV.")
        window = "Raw | Undistorted"
        win_w, win_h = 1920, 540

    print("Press 'q' to quit.")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, win_w, win_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        undistorted = cv2.undistort(frame, K, dist)
        und_panel = cv2.resize(undistorted, PANEL_SIZE)

        panels = [cv2.resize(frame, PANEL_SIZE), und_panel]

        if H is not None:
            flat = cv2.warpPerspective(und_panel, H, bev_size)
            panels.append(cv2.resize(flat, PANEL_SIZE))

        cv2.imshow(window, np.hstack(panels))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
