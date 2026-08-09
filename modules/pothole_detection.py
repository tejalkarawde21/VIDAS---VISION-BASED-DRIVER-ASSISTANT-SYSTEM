# modules/pothole_detection.py
from ultralytics import YOLO
import cv2

# Load trained pothole model
model = YOLO("runs/detect/train2/weights/best.pt")


def detect_potholes(frame):
    """
    Returns (annotated_frame, pothole_count).
    Uses ORANGE boxes instead of red to avoid being picked up
    as red-colour false positives by the traffic light scanner.
    """
    output        = frame.copy()
    pothole_count = 0

    results = model.predict(
        source=frame,
        conf=0.25,          # slightly higher threshold — reduce noise
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── ORANGE box — not red, not green ──────────
            # BGR: (0, 165, 255) = orange
            BOX_COL   = (0, 140, 255)   # orange in BGR
            LABEL_COL = (0, 140, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COL, 2)

            label = f"Pothole {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # small pill background for label
            pad = 3
            lx1, ly1 = x1, max(0, y1 - th - 2*pad - 2)
            lx2, ly2 = x1 + tw + 2*pad, y1
            ov = output.copy()
            cv2.rectangle(ov, (lx1, ly1), (lx2, ly2), (20, 20, 20), -1)
            cv2.addWeighted(ov, 0.7, output, 0.3, 0, output)
            cv2.rectangle(output, (lx1, ly1), (lx1+2, ly2), BOX_COL, -1)

            cv2.putText(
                output, label,
                (lx1 + pad + 3, ly2 - pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, LABEL_COL, 1, cv2.LINE_AA
            )

            pothole_count += 1

    return output, pothole_count