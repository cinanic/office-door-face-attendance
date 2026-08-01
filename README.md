# Office Door Face Attendance

Recognizes employees at the door camera and logs Entrance/Exit events to a CSV.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want faster processing, install
`onnxruntime-gpu` instead of `onnxruntime`, and pass `ctx_id=0` (edit
`get_face_app` in both scripts, or just ask me and I'll parametrize it).

The first run will auto-download the InsightFace `buffalo_l` model
(~300MB) to `~/.insightface`.

## 2. Organize your employee data

```
employee_data/
    John Smith/
        photo1.jpg
        clip.mp4
    Jane Doe/
        photo1.png
        clip.mov
```

One folder per employee, folder name = exact name you want in the CSV.
Mix images and/or short videos freely — more angles/lighting = better
matching.

## 3. Enroll employees (build the face database)

```bash
python enroll_employees.py --data_dir employee_data --out employees_db.pkl
```

This produces `employees_db.pkl`, a compact file mapping each employee
name to an averaged face embedding.

## 4. Test your Wi-Fi camera connection first

Your camera's stream URL is hardcoded at the top of `test_camera.py`
and `door_recognition.py`:

```python
CAMERA_SOURCE = "http://192.168.1.204:81/stream"
```

Edit that line in both files if the camera's IP ever changes. Then
just run (no arguments needed — works with a plain Run/F5 in Spyder
too):

```bash
python test_camera.py
```

## 5. Run door recognition

```bash
# Uses CAMERA_SOURCE from the top of the file, with the live preview window
python door_recognition.py

# Run headless (no preview window)
python door_recognition.py --no-display

# Override the source for a one-off test, e.g. a recorded video or USB webcam
python door_recognition.py --source door_test.mp4 --db employees_db.pkl
python door_recognition.py --source 0
```

This continuously writes to `attendance_log.csv`:

| Employee   | Event    | Timestamp           |
|------------|----------|----------------------|
| John Smith | Entrance | 2026-07-26 09:02:11 |
| Jane Doe   | Entrance | 2026-07-26 09:05:44 |
| John Smith | Exit     | 2026-07-26 12:31:02 |

Unrecognized faces are shown as "Unknown" in the live display (if
`--display` is on) but are **not** written to the CSV — only enrolled
employees are logged. Say the word if you'd also like unknown visitors
logged to a separate CSV.

## Key settings to tune (top of `door_recognition.py`)

- `RECOGNITION_THRESHOLD` (default 0.45): similarity cutoff for a
  match. Raise it (e.g. 0.5–0.55) if you get false matches between
  different people; lower it if the same employee isn't being
  recognized consistently.
- `COOLDOWN_SECONDS` (default 15): how long a person must be out of
  frame before a new appearance counts as a fresh Entrance/Exit event,
  instead of just refreshing the same one (prevents someone lingering
  at the door from generating dozens of rows).
- `PROCESS_EVERY_N_FRAMES` (default 3): only run detection every N
  frames to save CPU. Lower for more responsiveness, raise if the feed
  lags.
- `CONSECUTIVE_MATCHES_REQUIRED` (default 4): how many processed
  frames in a row must agree on the same person before an event is
  logged. Filters out one-off misfires from a bad angle or a flicker
  of lighting — raise it for more certainty (slower to log), lower it
  for a snappier response.
- `MAX_FACES_ALLOWED` (default 1): if more faces than this show up in
  one frame, no event is logged for that frame at all — this avoids
  misattributing an entrance/exit when multiple people are at the door
  together. Raise this if groups regularly walk through together and
  you still want each person logged individually (accuracy trade-off:
  with several faces in frame it's easier to mismatch confidence
  scores between them).
- `RECONNECT_DELAY_SECONDS` (default 2): if the camera stream drops
  (common on Wi-Fi cameras), how long to wait before retrying the
  connection. The script keeps retrying indefinitely rather than
  exiting.

## About the Entrance/Exit logic

A single camera facing the door can't inherently tell direction. This
script uses the simplest reasonable convention: each employee's
appearances alternate Entrance → Exit → Entrance → ... in order,
**and this resets every calendar day** — the first detection of a new
day is always logged as an Entrance, regardless of how the previous
day ended. That way, if someone enters but never passes the camera
again that day (leaves through a different door, camera misses the
exit, etc.), the next day still starts fresh instead of incorrectly
logging an Exit.

Within a day, this works well if people reliably pass the camera once
per entrance and once per exit.

If your setup needs true directional detection (e.g., a line the
person crosses, and which side they're walking from/to), that's a
different, somewhat more involved approach using multi-frame motion
tracking — let me know and I can build that version instead.
