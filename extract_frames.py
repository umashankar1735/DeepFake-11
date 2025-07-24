import os
import json
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
metadata_path = BASE_DIR / "data" / "metadata.json"
dataset_dir = BASE_DIR / "data" / "train_sample_videos"
output_dir = BASE_DIR / "data" / "extracted_frames"

if not metadata_path.exists():
    print(f"[✗] ERROR: metadata.json NOT found at {metadata_path}")
    exit(1)

os.makedirs(output_dir, exist_ok=True)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

for video_file, info in metadata.items():
    label = info['label']
    video_path = dataset_dir / video_file

    if not video_path.exists():
        print(f"[!] Missing: {video_file}")
        continue

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    success, frame = cap.read()
    cap.release()

    if success:
        filename = f"{Path(video_file).stem}_{label}.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)
        print(f"[✓] Saved: {output_path}")
    else:
        print(f"[!] Failed: {video_file}")
