#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles for serving static files
from fastapi.responses import FileResponse
from keras.models import load_model
import cv2
import numpy as np
import asyncio

try:
    model = load_model("emotion_model.h5")
    label_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    raise

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error loading Haarcascade file.")
    raise Exception("Haarcascade file not loaded successfully.")
print("Haarcascade loaded successfully.")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    try:
        return FileResponse("index.html")
    except Exception as e:
        return {"error": str(e)}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/real-time-emotion")
async def real_time_emotion(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion recognition.
    Captures webcam feed, detects faces, and predicts emotions using a pre-trained model.
    """
    print("WebSocket /real-time-emotion accessed.")
    await websocket.accept()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Unable to access the webcam."})
        await websocket.close()
        print("WebSocket closed: Unable to access webcam.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            emotions = []
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.reshape(1, 48, 48, 1) / 255.0

                # Predict emotion
                predictions = model.predict(face)
                emotion_index = np.argmax(predictions)
                emotion_label = label_classes[emotion_index]
                confidence = predictions[0][emotion_index]

                emotions.append({
                    "emotion": emotion_label,
                    "confidence": float(confidence),
                    "coordinates": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })

            # Send emotions via WebSocket
            await websocket.send_json({"emotions": emotions})

            # Limit the frame rate
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        await websocket.send_json({"error": str(e)})

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("WebSocket connection closed.")
