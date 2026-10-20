# VIDAS – Vision Based Driver Assistant System 🚗👁️

VIDAS is a real-time driver assistance system that uses computer vision to detect road hazards and conditions, then alerts the driver through voice notifications — helping improve road safety, especially in low-visibility or high-risk driving situations.

## 🔍 Features

- **Fog Detection** – Identifies dense fog conditions and warns the driver to reduce speed
- **Lane Detection** – Detects lane markings to help keep the vehicle on track
- **Object Detection** – Identifies vehicles, pedestrians, and obstacles on the road using YOLO
- **Pothole Detection** – Flags potholes ahead using a custom-trained YOLO model
- **Traffic Light Detection** – Recognizes traffic signal states (red/yellow/green)
- **Voice Alerts** – Converts detections into real-time spoken alerts, so drivers don't need to look away from the road

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 / YOLO11** (Ultralytics) – object, pothole, and segmentation models
- **OpenCV** – image and video processing
- **Text-to-Speech Engine** – for real-time voice alerts

## 🧠 How It Works

Each frame from the camera feed is passed through the relevant detection modules. Detected conditions (e.g., dense fog, a pothole ahead, an approaching vehicle) are converted into alert messages, which `voice_engine.py` then reads out to the driver in real time — enabling hands-free, eyes-on-the-road hazard awareness.

## 🚀 Future Scope

- Integration with real vehicle hardware / dashcam feeds
- Mobile app companion for alert history and trip logs
- Expanded dataset training for better accuracy in varied weather conditions

## 👩‍💻 Author

**Tejal Karawde**
B.Tech Data Science, Usha Mittal Institute of Technology (SNDT Women's University)

