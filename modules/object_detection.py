# modules/object_detection.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLOv8n model (will download yolov8n.pt on first run)
# Keep this global so model loads only once
MODEL_NAME = "yolov8n.pt"
model = YOLO(MODEL_NAME)

# Choose target classes you care about (COCO names)
TARGET_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle", "person", "traffic light"}

def detect_objects(frame, conf=0.25):
    """
    Input: BGR frame
    Output: annotated_frame, detections_list
    detections_list: list of dicts {'name','conf','box':(x1,y1,x2,y2)}
    """
    annotated = frame.copy()
    detections = []

    # Run YOLOv8 prediction (returns a Results object list)
    results = model.predict(frame, conf=conf, imgsz=640, verbose=False)
    if len(results) == 0:
        return annotated, detections

    res = results[0]  # first result for this frame
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        return annotated, detections

    names = model.names  # dict id->name

    # extract numpy arrays
    xyxy = boxes.xyxy.cpu().numpy()       # (n,4)
    confs = boxes.conf.cpu().numpy()     # (n,)
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # (n,)

    for box, score, cid in zip(xyxy, confs, cls_ids):
        name = names.get(cid, str(cid))
        x1, y1, x2, y2 = map(int, box)
        detections.append({'name': name, 'conf': float(score), 'box': (x1, y1, x2, y2)})
        # draw only target classes to avoid clutter
        if name in TARGET_CLASSES:
            color = (255, 0, 0) if name in ("car","truck","bus","motorbike") else (0,255,255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{name} {score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return annotated, detections
