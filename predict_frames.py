import os
import torch
import torch.nn as nn
from torchvision import transforms as T, models
from PIL import Image
import cv2
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model.pth"

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

def extract_frames(video_path, frame_step=30, max_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success:
            frame_path = BASE_DIR / f"temp_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        return prediction.item(), confidence.item()

def predict_video(video_path):
    frame_paths = extract_frames(video_path, frame_step=30)
    predictions = []

    for path in frame_paths:
        label, _ = predict_image(path)
        predictions.append(label)
        os.remove(path)

    if not predictions:
        print("❌ No frames extracted for prediction.")
        return

    count = Counter(predictions)
    majority_label, majority_count = count.most_common(1)[0]
    confidence = majority_count / len(predictions)

    label_str = "FAKE" if majority_label == 1 else "REAL"
    print(f"🎯 Majority Prediction: {label_str} ({confidence * 100:.2f}% confidence)")

# Test
# predict_video(BASE_DIR / "test_video.mp4")
