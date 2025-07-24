import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import shutil
import pickle
from pathlib import Path

# === 1. Setup base dir and model path ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model.pkl"

# === 2. Load the .pkl model ===
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
model.eval()

# === 3. Transform for input images ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === 4. FastAPI app setup ===
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your React app URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 5. Frame extraction from uploaded video ===
def extract_frame(video_path, frame_number=30):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    if success:
        frame_path = BASE_DIR / "temp_frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    return None

# === 6. Predict using model ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return ("FAKE" if prediction == 1 else "REAL", confidence)

# === 7. Upload API endpoint ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_video_dir = BASE_DIR / "temp_videos"
    temp_video_dir.mkdir(exist_ok=True)
    
    video_path = temp_video_dir / file.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frame_path = extract_frame(video_path)
    if frame_path:
        label, confidence = predict_image(frame_path)
        os.remove(frame_path)
        os.remove(video_path)
        return {"label": label, "confidence": round(confidence * 100, 2)}
    else:
        return {"error": "Frame extraction failed"}

# === 8. Run server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
