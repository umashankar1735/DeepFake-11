import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import cv2
from collections import Counter
from pathlib import Path

# === 1. Setup Paths ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model.pth"
VIDEO_FOLDER = BASE_DIR / "test_videos"
TEMP_FRAME_DIR = BASE_DIR / "temp_frames"
TEMP_FRAME_DIR.mkdir(exist_ok=True)

# === 2. Load the model ===
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === 3. Define image transform ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === 4. Extract frames from video ===
def extract_frames(video_path, frame_interval=30, max_frames=10):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0
    while count < total_frames and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        if not success:
            break
        frame_path = TEMP_FRAME_DIR / f"frame_{count}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        count += frame_interval
    cap.release()
    return frames

# === 5. Predict a single frame ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_class].item() * 100
    return pred_class, confidence

# === 6. Predict full video ===
def predict_video(video_path):
    frames = extract_frames(video_path)
    predictions = []
    for frame_path in frames:
        label, confidence = predict_image(frame_path)
        predictions.append(label)
        frame_path.unlink()  # Delete after use
    if not predictions:
        return "❌ No valid frames", 0.0
    majority = Counter(predictions).most_common(1)[0]
    label = "FAKE" if majority[0] == 1 else "REAL"
    confidence = (majority[1] / len(predictions)) * 100
    return label, confidence

# === 7. Run on all videos in the folder ===
if not VIDEO_FOLDER.exists():
    print("❌ Test video folder not found!")
else:
    for filename in os.listdir(VIDEO_FOLDER):
        if filename.lower().endswith(".mp4"):
            video_path = VIDEO_FOLDER / filename
            label, confidence = predict_video(video_path)
            print(f"🎬 {filename}: {label} ({confidence:.2f}% confidence)")
