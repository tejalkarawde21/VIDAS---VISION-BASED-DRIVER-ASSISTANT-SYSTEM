# modules/lane_detection.py
import cv2
import numpy as np

def detect_lanes(frame, debug=False):
    """
    Input: BGR frame
    Output: (annotated_frame, direction_str, offset_pixels)
    direction_str: "Left" / "Right" / "Straight"
    offset_pixels: lane_center_x - frame_center_x (positive -> lane center is to the right)
    """
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    # Region of interest - lower part of image
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (w, int(h * 0.6)),
        (0, int(h * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)

    # Hough lines
    lines = cv2.HoughLinesP(cropped, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=40, maxLineGap=150)

    left_x_bottoms = []
    right_x_bottoms = []
    line_img = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Avoid near-horizontal lines
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            # Discard nearly horizontal lines
            if abs(slope) < 0.3:
                continue

            # compute x coordinate at bottom (y = h)
            # linear interpolation: x_at_h = x1 + (h - y1) * (x2-x1)/(y2-y1)
            if (y2 - y1) != 0:
                x_at_bottom = int(x1 + (h - y1) * (x2 - x1) / (y2 - y1))
            else:
                x_at_bottom = x1

            # classify left or right based on slope sign
            if slope < 0:
                left_x_bottoms.append(x_at_bottom)
            else:
                right_x_bottoms.append(x_at_bottom)

            # draw raw line for visualization
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Average bottoms to get lane edges
    left_x = np.mean(left_x_bottoms).astype(int) if len(left_x_bottoms) > 0 else None
    right_x = np.mean(right_x_bottoms).astype(int) if len(right_x_bottoms) > 0 else None

    lane_center_x = None
    direction = "Unknown"
    offset = 0
    if left_x is not None and right_x is not None:
        lane_center_x = int((left_x + right_x) / 2)
        frame_center = w // 2
        offset = lane_center_x - frame_center
        # threshold in pixels for decision (tweakable)
        thresh = max(30, w * 0.03)
        if offset < -thresh:
            direction = "Turn Left"
        elif offset > thresh:
            direction = "Turn Right"
        else:
            direction = "Go Straight"

        # Draw lane center and frame center for debugging
        cv2.line(line_img, (lane_center_x, h), (lane_center_x, int(h*0.6)), (255, 0, 0), 2)
        cv2.line(line_img, (frame_center, h), (frame_center, int(h*0.6)), (0, 0, 255), 2)
    else:
        direction = "Unknown"

    # Overlay lane lines on original
    annotated = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)

    # small status text
    #cv2.putText(annotated, f"Turn: {direction}", (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if lane_center_x is not None:
        cv2.putText(annotated, f"Offset: {offset}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    return annotated, direction, offset
