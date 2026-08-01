"""
enroll_employees.py

Builds a face-embedding database from employee photos and/or short videos.

Expected folder layout:

    employee_data/
        John Smith/
            photo1.jpg
            photo2.jpg
            clip.mp4
        Jane Doe/
            photo1.png
            clip.mov
        ...

- One subfolder per employee, named exactly how you want their name to
  appear in the CSV log.
- Inside each subfolder, put any mix of images (.jpg/.jpeg/.png) and/or
  videos (.mp4/.mov/.avi). Videos are sampled every N frames (see
  FRAME_SAMPLE_STEP below) to pull additional face examples.

Output:
    employees_db.pkl  -- a dict {employee_name: mean_embedding (512-d np.array)}

Usage:
    python enroll_employees.py --data_dir employee_data --out employees_db.pkl
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
FRAME_SAMPLE_STEP = 10  # take one frame every N frames from videos


def get_face_app(ctx_id=-1):
    """ctx_id=-1 uses CPU. Set to 0 (or your GPU id) if you have a CUDA GPU
    and installed onnxruntime-gpu instead of onnxruntime."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def embeddings_from_image(app, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [warn] could not read image: {image_path}")
        return []
    faces = app.get(img)
    if not faces:
        print(f"  [warn] no face found in: {image_path}")
        return []
    # If multiple faces are in one enrollment photo, use the largest one
    # (assumed to be the intended subject).
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return [faces[0].normed_embedding]


def embeddings_from_video(app, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [warn] could not open video: {video_path}")
        return []

    embeddings = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLE_STEP == 0:
            faces = app.get(frame)
            if faces:
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                embeddings.append(faces[0].normed_embedding)
        frame_idx += 1
    cap.release()
    print(f"  extracted {len(embeddings)} usable frames from {os.path.basename(video_path)}")
    return embeddings


def enroll(data_dir, out_path):
    app = get_face_app()
    db = {}

    employee_dirs = [d for d in sorted(os.listdir(data_dir))
                      if os.path.isdir(os.path.join(data_dir, d))]

    if not employee_dirs:
        print(f"No employee subfolders found in {data_dir}")
        sys.exit(1)

    for name in employee_dirs:
        person_dir = os.path.join(data_dir, name)
        print(f"Enrolling: {name}")
        all_embeddings = []

        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in IMAGE_EXTS:
                all_embeddings.extend(embeddings_from_image(app, fpath))
            elif ext in VIDEO_EXTS:
                all_embeddings.extend(embeddings_from_video(app, fpath))

        if not all_embeddings:
            print(f"  [warn] no usable face data for {name}, skipping")
            continue

        mean_embedding = np.mean(np.stack(all_embeddings), axis=0)
        # re-normalize after averaging
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        db[name] = mean_embedding
        print(f"  -> {len(all_embeddings)} face samples used, embedding stored")

    with open(out_path, "wb") as f:
        pickle.dump(db, f)

    print(f"\nSaved database for {len(db)} employees to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="employee_data",
                         help="Folder containing one subfolder per employee")
    parser.add_argument("--out", default="employees_db.pkl",
                         help="Output path for the embedding database")
    args = parser.parse_args()
    enroll(args.data_dir, args.out)
