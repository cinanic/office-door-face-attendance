"""
door_recognition.py

Runs face recognition on a live camera feed (or a video file, for testing)
pointed at the office door, matches faces against the enrolled employee
database, and logs Entrance/Exit events to a CSV file.

Reliability features:
    - Reconnects automatically if the camera stream drops.
    - Requires CONSECUTIVE_MATCHES_REQUIRED consecutive processed frames
      to agree on the same person before logging an event -- filters out
      one-off misfires from a bad angle or a flicker of lighting.
    - Anti-tailgating: if more than MAX_FACES_ALLOWED faces are in frame
      at once, no event is logged for that frame (you can raise
      MAX_FACES_ALLOWED if your door is fine with people entering as a
      group and you just want everyone logged individually -- see note
      below the constant).

Entrance/Exit logic (since a single camera has no reliable "direction"
information on its own):
    - Each employee has a "last event" state (starts as None).
    - The FIRST detection of each calendar day is always logged as an
      Entrance, no matter how the previous day ended -- so if someone
      never passed the camera again after an entrance (e.g. left
      through a different door, or the camera missed it), the next
      day still starts fresh instead of incorrectly logging an Exit.
    - Within the same day, appearances alternate Entrance -> Exit ->
      Entrance -> ..., only counting a new appearance after the
      person has been absent for at least COOLDOWN_SECONDS.
    - State is saved to attendance_state.json so it survives restarts.

If you want *true* direction detection (e.g. using a virtual line the
person crosses, walking toward vs away from camera), that requires
tracking bounding-box motion across frames -- ask and I can add that
version instead.

Usage:
    # Wi-Fi / IP camera (e.g. ESP32-CAM style, MJPEG stream)
    python door_recognition.py --db employees_db.pkl --source "http://192.168.1.204:81/stream" --display

    # Live USB webcam (device 0)
    python door_recognition.py --db employees_db.pkl --source 0

    # RTSP camera
    python door_recognition.py --db employees_db.pkl --source "rtsp://user:pass@ip/stream"

    # Test on a video file
    python door_recognition.py --db employees_db.pkl --source test_video.mp4 --display
"""

import argparse
import csv
import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def open_capture(source):
    """Opens the video source exactly the way the working recognize.py
    script does: a plain cv2.VideoCapture(source) call, no test-read or
    reopen tricks. Works for a webcam index, a video file, an RTSP URL,
    or (per your other working script) an HTTP MJPEG stream URL."""
    if isinstance(source, str) and not source.startswith(("http", "rtsp")):
        try:
            source = int(source)
        except ValueError:
            pass
    return cv2.VideoCapture(source)


RECOGNITION_THRESHOLD = 0.45      # cosine similarity; raise for stricter matching
COOLDOWN_SECONDS = 15             # min seconds absent before a re-appearance counts as a new event
PROCESS_EVERY_N_FRAMES = 3        # skip frames for speed; raise if CPU-bound
CONSECUTIVE_MATCHES_REQUIRED = 4  # processed frames in a row that must agree before logging
MAX_FACES_ALLOWED = 1             # more faces than this in one frame -> skip logging (anti-tailgating).
                                   # Raise this if multiple employees regularly walk through
                                   # together and you want each still logged individually.
RECONNECT_DELAY_SECONDS = 2       # wait time before retrying a dropped camera stream


def get_face_app(ctx_id=-1):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def load_db(db_path):
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    names = list(db.keys())
    matrix = np.stack([db[n] for n in names])  # (N, 512)
    return names, matrix


def match_face(embedding, names, matrix, threshold=RECOGNITION_THRESHOLD):
    sims = matrix @ embedding  # cosine similarity since embeddings are normalized
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return names[best_idx], best_sim
    return None, best_sim


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {}


def save_state(state_path, state):
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def ensure_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Employee", "Event", "Timestamp"])


def log_event(csv_path, employee, event, ts):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([employee, event, ts])


def run(db_path, source, csv_path, state_path, display=False):
    names, matrix = load_db(db_path)
    print(f"Loaded {len(names)} employees: {', '.join(names)}")

    app = get_face_app()
    ensure_csv(csv_path)
    state = load_state(state_path)  # {employee_name: {"last_event": "Entrance"/"Exit", "last_seen": epoch_seconds}}
    match_streak = defaultdict(int)  # name -> consecutive processed frames matched

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return

    frame_idx = 0
    print("Running. Press Ctrl+C (or 'q' in the display window) to stop.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print(f"Camera stream dropped. Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
                cap.release()
                time.sleep(RECONNECT_DELAY_SECONDS)
                cap = open_capture(source)
                if not cap.isOpened():
                    print("Reconnect failed, will keep retrying...")
                continue

            frame_idx += 1
            now = time.time()

            if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                faces = app.get(frame)
                names_seen_this_frame = set()

                if len(faces) > MAX_FACES_ALLOWED:
                    # Anti-tailgating: too many people in frame at once to
                    # safely attribute an event to a specific person. Reset
                    # all streaks so nobody's partial progress carries over.
                    match_streak.clear()
                    for face in faces:
                        box = face.bbox.astype(int)
                        if display:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 200), 2)
                            cv2.putText(frame, "multiple people", (box[0], max(box[1] - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                else:
                    for face in faces:
                        name, sim = match_face(face.normed_embedding, names, matrix)
                        box = face.bbox.astype(int)

                        if name is None:
                            label = f"Unknown ({sim:.2f})"
                            color = (0, 0, 255)
                        else:
                            names_seen_this_frame.add(name)
                            match_streak[name] += 1
                            streak_ok = match_streak[name] >= CONSECUTIVE_MATCHES_REQUIRED

                            suffix = "" if streak_ok else f" [{match_streak[name]}/{CONSECUTIVE_MATCHES_REQUIRED}]"
                            label = f"{name} ({sim:.2f}){suffix}"
                            color = (0, 200, 0)

                            if streak_ok:
                                prev = state.get(name)
                                today_str = datetime.now().strftime("%Y-%m-%d")
                                is_new_day = prev is None or prev.get("last_date") != today_str

                                if is_new_day or (now - prev["last_seen"]) > COOLDOWN_SECONDS:
                                    if is_new_day:
                                        # First sighting of this person today -- always
                                        # an Entrance, regardless of how yesterday ended
                                        # (e.g. they never logged an Exit because they
                                        # left without passing the camera again).
                                        next_event = "Entrance"
                                    else:
                                        next_event = "Exit" if prev["last_event"] == "Entrance" else "Entrance"
                                    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    log_event(csv_path, name, next_event, ts_str)
                                    state[name] = {"last_event": next_event, "last_seen": now, "last_date": today_str}
                                    save_state(state_path, state)
                                    print(f"[{ts_str}] {name}: {next_event}")
                                else:
                                    # still within cooldown, just refresh last_seen so they
                                    # don't get double-logged while lingering at the door
                                    state[name]["last_seen"] = now

                        if display:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(frame, label, (box[0], max(box[1] - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Anyone not matched this processed frame breaks their streak
                # (blip, turned away, or left frame) so a later re-appearance
                # has to build the streak up again from zero.
                for known_name in list(match_streak.keys()):
                    if known_name not in names_seen_this_frame:
                        match_streak[known_name] = 0

            if display:
                cv2.imshow("Door camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # ---- EDIT THIS if your camera's address changes -------------------
    CAMERA_SOURCE = "http://192.168.1.204:81/stream"
    # ---------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="employees_db.pkl", help="Path to enrolled employee database")
    parser.add_argument("--source", default=CAMERA_SOURCE,
                         help="Camera index (e.g. 0), video file path, RTSP URL, or HTTP MJPEG stream URL")
    parser.add_argument("--csv", default="attendance_log.csv", help="Output CSV path")
    parser.add_argument("--state", default="attendance_state.json", help="State file for entrance/exit toggling")
    parser.add_argument("--no-display", action="store_true", help="Run headless, without the live preview window")
    args = parser.parse_args()

    # Works with a plain Run/F5 in Spyder (no arguments needed) since every
    # argument above has a default -- CAMERA_SOURCE is used automatically.
    run(args.db, args.source, args.csv, args.state, display=not args.no_display)
