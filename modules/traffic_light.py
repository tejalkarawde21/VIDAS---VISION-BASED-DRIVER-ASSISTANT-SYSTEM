# modules/traffic_light.py
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.18   # minimum YOLO confidence to analyse a box
MIN_PIXELS     = 40     # minimum coloured pixels inside box to declare
VOTE_RATIO     = 1.4    # winner must be this × runner-up


# ──────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────
def _analyse_crop(crop: np.ndarray):
    """
    Returns (red_px, green_px, yellow_px).
    CLAHE + wide HSV so nighttime LED signals are detected.
    """
    if crop is None or crop.size == 0:
        return 0, 0, 0

    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # boost V channel so dim lights become detectable
    h, s, v = cv2.split(hsv)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v       = clahe.apply(v)
    hsv     = cv2.merge([h, s, v])

    # RED — two hue bands + washed-out overexposed LED red
    r1 = cv2.inRange(hsv, np.array([0,   60,  80]), np.array([12,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 60,  80]), np.array([180, 255, 255]))
    r3 = cv2.inRange(hsv, np.array([0,   10, 185]), np.array([15,  100, 255]))
    r4 = cv2.inRange(hsv, np.array([160, 10, 185]), np.array([180, 100, 255]))
    red_mask = r1 | r2 | r3 | r4

    # GREEN — standard + teal-green LEDs
    g1 = cv2.inRange(hsv, np.array([40,  60,  80]), np.array([85,  255, 255]))
    g2 = cv2.inRange(hsv, np.array([85,  40, 120]), np.array([100, 255, 255]))
    green_mask = g1 | g2

    # YELLOW
    y1 = cv2.inRange(hsv, np.array([22, 120, 150]), np.array([35,  255, 255]))
    yellow_mask = y1

    k = np.ones((2, 2), np.uint8)
    red_mask    = cv2.morphologyEx(red_mask,    cv2.MORPH_OPEN, k)
    green_mask  = cv2.morphologyEx(green_mask,  cv2.MORPH_OPEN, k)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, k)

    return (cv2.countNonZero(red_mask),
            cv2.countNonZero(green_mask),
            cv2.countNonZero(yellow_mask))


def _vote(red, green, yellow):
    counts = {"RED": red, "GREEN": green, "YELLOW": yellow}
    best   = max(counts, key=counts.get)
    bval   = counts[best]
    if bval < MIN_PIXELS:
        return "UNKNOWN"
    others = [v for k, v in counts.items() if k != best]
    second = max(others) if others else 0
    if second > 0 and bval < second * VOTE_RATIO:
        return "UNKNOWN"
    return best


def _top_half(crop):
    """Top 60% of crop — where the lit bulb usually sits."""
    if crop is None or crop.size == 0:
        return crop
    return crop[:max(1, int(crop.shape[0] * 0.60)), :]


def _draw_box(img, x1, y1, x2, y2, status, conf):
    col_map = {
        "RED":     (0,   0,   220),
        "GREEN":   (0,   210,  50),
        "YELLOW":  (0,   210, 230),
        "UNKNOWN": (110, 110, 110),
    }
    col   = col_map.get(status, (110, 110, 110))
    label = f"TL:{status} {conf:.2f}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    sc, th = 0.44, 1
    (tw, lh), _ = cv2.getTextSize(label, font, sc, th)
    pad = 3
    lx1 = x1
    ly1 = max(0, y1 - lh - 2*pad - 2)
    lx2 = x1 + tw + 2*pad
    ly2 = y1
    ov  = img.copy()
    cv2.rectangle(ov, (lx1, ly1), (lx2, ly2), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (lx1, ly1), (lx1+2, ly2), col, -1)
    cv2.putText(img, label, (lx1+pad+3, ly2-pad-1),
                font, sc, col, th, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)


# ──────────────────────────────────────────────────────────
# PUBLIC API
# detections: list of dicts {'name', 'conf', 'box':(x1,y1,x2,y2)}
#
# KEY DESIGN DECISION:
#   NO ROI / colour fallback.
#   If YOLO does not detect a traffic light box → return UNKNOWN.
#   This prevents vehicle taillights, pothole boxes, and other red
#   objects from being misclassified as RED signals.
# ──────────────────────────────────────────────────────────
def detect_traffic_light_color(frame: np.ndarray, detections: list):
    """
    Returns (annotated_frame, status)
    status: "RED" | "GREEN" | "YELLOW" | "UNKNOWN"
    UNKNOWN means no traffic light was detected — not a signal state.
    """
    annotated = frame.copy()
    H, W      = frame.shape[:2]

    # ── Step 1: collect all YOLO traffic light boxes ──────
    tl_dets = [
        d for d in detections
        if d.get("name") == "traffic light"
    ]

    # If YOLO found absolutely no traffic light → UNKNOWN immediately
    # Do NOT scan the frame for colours — too many false positives
    if not tl_dets:
        return annotated, "UNKNOWN"

    # ── Step 2: filter by confidence ─────────────────────
    confident_dets = [d for d in tl_dets if d.get("conf", 0) >= CONF_THRESHOLD]

    # If all boxes are below threshold, use best one but require
    # higher pixel count to accept result
    if not confident_dets:
        best = max(tl_dets, key=lambda d: d["conf"])
        x1, y1, x2, y2 = best["box"]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)
        crop = frame[y1:y2, x1:x2]
        top  = _top_half(crop)
        r1, g1, yc1 = _analyse_crop(crop)
        r2, g2, yc2 = _analyse_crop(top)
        # Require more pixels when confidence is low
        r = r1 + r2*2; g = g1 + g2*2; y = yc1 + yc2*2
        mx = max(r, g, y)
        if mx >= MIN_PIXELS * 2:      # stricter threshold
            result = _vote(r, g, y)
            _draw_box(annotated, x1, y1, x2, y2, result, best["conf"])
            return annotated, result
        return annotated, "UNKNOWN"

    # ── Step 3: confidence-weighted voting across all boxes
    confident_dets.sort(key=lambda d: d["conf"], reverse=True)
    colour_votes = {"RED": 0.0, "GREEN": 0.0,
                    "YELLOW": 0.0, "UNKNOWN": 0.0}

    for d in confident_dets:
        x1, y1, x2, y2 = d["box"]
        conf = d["conf"]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)

        crop     = frame[y1:y2, x1:x2]
        top_crop = _top_half(crop)

        r1, g1, y1c = _analyse_crop(crop)
        r2, g2, y2c = _analyse_crop(top_crop)
        r = r1 + r2*2; g = g1 + g2*2; y = y1c + y2c*2

        result = _vote(r, g, y)
        colour_votes[result] += conf
        _draw_box(annotated, x1, y1, x2, y2, result, conf)

    # ── Step 4: pick final winner ─────────────────────────
    meaningful = {k: v for k, v in colour_votes.items()
                  if k != "UNKNOWN" and v > 0}

    if meaningful:
        return annotated, max(meaningful, key=meaningful.get)

    # YOLO found boxes but colour analysis was inconclusive → UNKNOWN
    return annotated, "UNKNOWN"