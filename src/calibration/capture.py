import cv2
import os

CAMERA_INDEX = 4
RESOLUTION = (3840, 2160)
SAVE_DIR = "validation_images"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    if not cap.isOpened():
        print(f"Could not open camera at index {CAMERA_INDEX}")
        return

    count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
    print(f"Existing images: {count}")
    print("SPACE = capture  |  q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"Captured: {count}  |  SPACE=capture  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Calibration Images", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = os.path.join(SAVE_DIR, f"calib_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")
            count += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
