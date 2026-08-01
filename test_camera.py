"""
test_camera.py

Quick standalone check that you can pull frames from your Wi-Fi door
camera. No command-line arguments needed -- just press Run (F5) in
Spyder. Edit CAMERA_SOURCE below if your camera's IP changes.

This has ZERO dependencies beyond opencv-python -- it does not import
door_recognition.py or insightface, so if this fails, the problem is
the camera/network, not the face recognition stack.

Opens the camera exactly the way your working recognize.py script
does: cv2.VideoCapture(CAMERA_SOURCE), nothing else.
"""

import time

import cv2

# ---- EDIT THIS if your camera's address changes -------------------
CAMERA_SOURCE = "http://192.168.1.204:81/stream"
# ---------------------------------------------------------------------

DISPLAY = True       # set False to skip the preview window (headless run)
TEST_SECONDS = 15    # how long to read frames for


def main(source=CAMERA_SOURCE, display=DISPLAY, seconds=TEST_SECONDS):
    print(f"Connecting to: {source}")
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        print("FAILED: cv2.VideoCapture could not open this source.")
        print("This is the exact same call your working recognize.py uses, so if")
        print("it fails here, it's a network/URL/camera issue, not a code issue:")
        print(f"  1. Open this exact URL in a browser or VLC on THIS machine: {source}")
        print("     If it doesn't load there either, the camera isn't reachable right")
        print("     now (wrong/changed IP, camera off, different Wi-Fi, firewall).")
        print("  2. Confirm this computer and the camera are on the same network.")
        return

    print("Connected. Reading frames for a bit...")
    start = time.time()
    frame_count = 0
    frame = None

    while time.time() - start < seconds:
        ok, frame = video.read()
        if not ok:
            print("Stream stopped returning frames.")
            break
        frame_count += 1

        if display:
            cv2.imshow("Camera test - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nReceived {frame_count} frames in {elapsed:.1f}s (~{fps:.1f} fps)")
    if frame_count > 0 and frame is not None:
        h, w = frame.shape[:2]
        print(f"Frame size: {w}x{h}")
        print("Camera connection works. You can proceed to door_recognition.py.")
    else:
        print("No frames received.")

    video.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
