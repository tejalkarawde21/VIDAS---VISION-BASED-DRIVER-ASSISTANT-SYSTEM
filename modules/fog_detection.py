# modules/fog_detection.py

import cv2
import numpy as np

def detect_fog(frame):
    """
    Returns:
    fog_status (str): CLEAR / LIGHT FOG / DENSE FOG
    fog_score (float): contrast score
    """

    img = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast (visibility indicator)
    contrast = gray.std()

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Decision logic (realistic thresholds)
    if contrast < 60 and edge_density < 0.015:
        fog_status = "DENSE FOG"
    elif contrast < 120:
        fog_status = "LIGHT FOG"
    else:
        fog_status = "CLEAR"

    return fog_status, round(contrast, 1)
