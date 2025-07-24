# 🚀 DeepFakeDetect++

An AI-powered web application that detects deepfake or manipulated media using cutting-edge deep learning and computer vision techniques.

## 🔍 Overview

**DeepFakeDetect++** is a full-stack application that allows users to upload videos/images and get real-time predictions on whether the media is real or fake. It leverages a custom-trained deep learning model on top of ResNet18 and integrates with a React frontend and FastAPI backend.

## 🎯 Key Features

- 🎥 Upload media (image/video) via an interactive UI  
- 🧠 Predict deepfake content with a trained AI model  
- 📊 Display prediction results with confidence scores  
- 💡 Clean and responsive React interface  
- 🚀 FastAPI backend for robust performance  
- 🔐 Handles real-time inference with `pickle` model integration

## 🛠️ Tech Stack

- **Frontend**: React.js, Tailwind CSS  
- **Backend**: FastAPI, Python  
- **Model**: ResNet18-based classifier  
- **Deployment Ready**: Backend and model structured for Streamlit or Uvicorn-based deployment

## 📁 Project Structure

```bash
DeepFakeDetect++
├── front-end
│   ├── public
│   ├── src
│   │   ├── components
│   │   ├── pages
│   │   └── App.js
│   └── package.json
├── back-end
│   ├── main.py
│   ├── model.pkl
│   └── requirements.txt
└── README.md'
```

## 🧪 How It Works

The user uploads a video/image through the frontend.
The file is sent via a POST request to the FastAPI /predict/ endpoint.
The backend processes the file, runs it through the trained model, and returns:
label: Real or Fake
confidence: Prediction confidence in %

## 🖼️ UI Sneak Peek


## 🏗️ Installation & Setup

**Backend**
- cd back-end
- python -m venv venv
- venv\Scripts\activate        # For Windows
- pip install -r requirements.txt
- uvicorn main:app --reload

**Frontend**
- cd front-end
- npm install
- npm start

## 🤝 Team
- K.Sai Uma Shankar
- G. Ram Trilok
- R. Navaneeth
- G. Manideep
