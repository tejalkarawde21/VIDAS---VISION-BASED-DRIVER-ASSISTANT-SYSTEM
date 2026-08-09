import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import random
import time
import math
from collections import deque

# Import perception modules
from modules.lane_detection import detect_lanes
from modules.object_detection import detect_objects
from modules.traffic_light import detect_traffic_light_color
from modules.pothole_detection import detect_potholes
from modules.fog_detection import detect_fog

# Import centralized voice system
from voice_engine import VoiceEngine

voice_engine   = VoiceEngine()
prev_tl_status = "UNKNOWN"
tl_history     = deque(maxlen=6)   # smooth over 6 frames

# ──────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────
BG         = "#0a0a0a"
BG_CARD    = "#111111"
BG_BTN     = "#141414"
AMBER      = "#FFB300"
WHITE      = "#F5F5F5"
GREY       = "#3a3a3a"
GREY_LIGHT = "#777777"
RED        = "#FF3B3B"
GREEN      = "#39FF14"
BLUE_ICE   = "#89CFF0"


# ──────────────────────────────────────────────────────────
# HUD HELPER
# ──────────────────────────────────────────────────────────
def _hud_row(frame, label, value, y,
             label_col=(160, 160, 160), value_col=(245, 245, 245)):
    cv2.putText(frame, label, (22, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_col, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (22, y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, value_col, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────
# PROCESS FRAME
# ──────────────────────────────────────────────────────────
def process_frame(frame, current_speed=0, session_start=0, is_image=False):
    global prev_tl_status

    frame = cv2.resize(frame, (960, 540))

    # keep a clean copy for TL colour sampling
    # (done BEFORE any overlay is drawn on the frame)
    clean_frame = frame.copy()

    # 1. LANE
    lane_frame, direction, offset = detect_lanes(frame)
    if abs(offset) < 25:
        direction = "STRAIGHT"

    # 2. OBJECTS — detect on lane frame
    obj_frame, detections = detect_objects(lane_frame, conf=0.20)

    # 3. TRAFFIC LIGHT — analyse on CLEAN frame
    #    This prevents pothole boxes / HUD overlays from polluting colour scan
    final_frame, tl_status = detect_traffic_light_color(clean_frame, detections)

    # Merge object detection visuals into final_frame
    # (obj_frame has the coloured bounding boxes for vehicles/persons)
    final_frame = cv2.addWeighted(final_frame, 0.5, obj_frame, 0.5, 0)

    # ── temporal smoothing on TL status ──────────────────
    tl_history.append(tl_status)
    counts = {}
    for s in tl_history:
        counts[s] = counts.get(s, 0) + 1
    tl_status = max(counts, key=counts.get)

    # ── parse detections ─────────────────────────────────
    pedestrian_count = 0
    vehicle_count    = 0
    emergency_stop   = False
    slow_down        = False
    close_pedestrian = False

    for det in detections:
        try:
            name     = det.get("name", "")
            box      = det.get("box", (0, 0, 0, 0))
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)

            if name == "person":
                pedestrian_count += 1
                if y2 > 460:
                    emergency_stop   = True
                    close_pedestrian = True

            if name in ("car", "truck", "bus", "motorbike"):
                vehicle_count += 1
                if box_area > (960 * 540 * 0.18):
                    slow_down = True
        except Exception:
            continue

    tl_status = tl_status.upper()

    # Only set emergency_stop from traffic light when a real signal is found
    if tl_status == "RED":
        emergency_stop = True
    elif tl_status == "YELLOW":
        slow_down = True

    # 4. FOG + POTHOLE
    fog_status, fog_score = detect_fog(final_frame)
    fog_status_upper      = fog_status.upper()
    if "DENSE" in fog_status_upper:
        slow_down = True

    final_frame, pothole_count = detect_potholes(final_frame)

    # ── 5. ALERT PRIORITY ────────────────────────────────
    # IMPORTANT: UNKNOWN traffic light means no signal detected
    # so we should NOT say "stop" — instead fall through to other alerts
    alert_message      = None
    alert_is_emergency = False

    # GREEN transition alert
    if tl_status == "GREEN" and prev_tl_status == "RED":
        alert_message = "Signal is green. Proceed when safe."

    # Pedestrian emergency
    elif close_pedestrian:
        if tl_status == "GREEN":
            alert_message = "Pedestrian crossing. Proceed carefully."
        else:
            if tl_status == "RED":
                alert_message = "RED Signal. Stop immediately."
                alert_is_emergency = True

    # RED signal — only when YOLO actually detected a traffic light
    elif tl_status == "RED":
        alert_message      = "Red signal. Stop the vehicle."
        alert_is_emergency = True

    # YELLOW signal
    elif tl_status == "YELLOW":
        alert_message = "Yellow signal. Slow down."

    # Dense fog
    elif "DENSE" in fog_status_upper:
        alert_message = "Dense fog ahead. Reduce speed."

    # Pothole — fires when no traffic signal is present (UNKNOWN or GREEN)
    elif pothole_count > 0:
        alert_message = "Pothole detected. Slow down."

    # Heavy traffic
    elif vehicle_count >= 3:
        alert_message = "Heavy traffic ahead. Drive carefully."

    # Many pedestrians
    elif pedestrian_count > 5:
        alert_message = "Crowded area ahead. Slow down."

    # Single vehicle close
    elif slow_down and vehicle_count > 0:
        alert_message = "Vehicle ahead. Reduce speed."

    # 6. VOICE
    if alert_message:
        voice_engine.speak(alert_message)

    # ── 7. HUD ───────────────────────────────────────────
    H, W = final_frame.shape[:2]

    # Panel bottom-left — does not cover upper area where signals appear
    panel_w, panel_h = 190, 185
    px, py = 10, H - panel_h - 10
    overlay = final_frame.copy()
    cv2.rectangle(overlay, (px, py),
                  (px + panel_w, py + panel_h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.52, final_frame, 0.48, 0, final_frame)

    # amber top accent line
    cv2.line(final_frame, (px, py), (px + panel_w, py),
             (0, 179, 255), 2)

    # VIDAS watermark
    cv2.putText(final_frame, "VIDAS", (W - 88, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (55, 55, 55), 1, cv2.LINE_AA)

    # traffic colour dot
    tl_dot = {
        "RED":     (0,   0,   220),
        "GREEN":   (0,   210,  50),
        "YELLOW":  (0,   200, 230),
        "UNKNOWN": (100, 100, 100),
    }.get(tl_status, (100, 100, 100))
    cv2.circle(final_frame,
               (px + panel_w - 18, py + 83), 8, tl_dot, -1)
    cv2.circle(final_frame,
               (px + panel_w - 18, py + 83), 8, (255, 255, 255), 1)

    # HUD rows
    turn_col = (0, 215, 255) if direction != "STRAIGHT" else (200, 200, 200)
    _hud_row(final_frame, "DIRECTION",  direction,
             py + 40,  value_col=turn_col)
    _hud_row(final_frame, "TRAFFIC",    tl_status,
             py + 80,  value_col=tl_dot)
    fog_col = (60, 120, 255) if "FOG" in fog_status_upper else (185, 185, 185)
    _hud_row(final_frame, "VISIBILITY", fog_status,
             py + 120, value_col=fog_col)
    ph_col = (0, 140, 255) if pothole_count > 0 else (185, 185, 185)
    _hud_row(final_frame, "POTHOLES",   str(pothole_count),
             py + 160, value_col=ph_col)

    # Alert banner
    if alert_message:
        bw = 460; bh = 38
        bx = (W - bw) // 2
        by = H - bh - 10
        b_bg     = (0, 0, 140)  if alert_is_emergency else (0, 85, 0)
        b_border = (0, 0, 220)  if alert_is_emergency else (0, 180, 0)
        ov2 = final_frame.copy()
        cv2.rectangle(ov2, (bx, by), (bx+bw, by+bh), b_bg, -1)
        cv2.addWeighted(ov2, 0.78, final_frame, 0.22, 0, final_frame)
        cv2.rectangle(final_frame, (bx, by), (bx+bw, by+bh), b_border, 1)
        cv2.circle(final_frame, (bx+16, by+bh//2), 5, b_border, -1)
        font = cv2.FONT_HERSHEY_DUPLEX; sc = 0.58
        (tw, _), _ = cv2.getTextSize(alert_message, font, sc, 1)
        tx = bx + (bw - tw)//2 + 10
        cv2.putText(final_frame, alert_message,
                    (tx, by+bh-10), font, sc,
                    (240, 240, 240), 1, cv2.LINE_AA)

    voice_engine.update_cycle()
    prev_tl_status = tl_status
    return final_frame


# ──────────────────────────────────────────────────────────
# UNIFIED FILE DIALOG
# ──────────────────────────────────────────────────────────
ALL_TYPES = [
    ("All Supported Files",
     "*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp *.webp"),
    ("Video Files",  "*.mp4 *.avi *.mov"),
    ("Image Files",  "*.jpg *.jpeg *.png *.bmp *.webp"),
]
VIDEO_EXT = {".mp4", ".avi", ".mov"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _open_file():
    import os
    path = filedialog.askopenfilename(filetypes=ALL_TYPES)
    if not path:
        return None, None
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXT: return path, "video"
    if ext in IMAGE_EXT: return path, "image"
    return path, None


def analyze_video(path=None):
    global voice_engine
    voice_engine.shutdown()
    voice_engine = VoiceEngine()
    if path is None:
        path, kind = _open_file()
        if not path: return
        if kind == "image":
            analyze_image(path); return
    cap = cv2.VideoCapture(path)
    cv2.namedWindow("VIDAS  ·  LIVE PERCEPTION", cv2.WINDOW_NORMAL)
    session_start = time.time()
    frame_count   = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 2 != 0: continue
        output = process_frame(frame, 0, session_start)
        cv2.imshow("VIDAS  ·  LIVE PERCEPTION", output)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()


def analyze_image(path=None):
    global voice_engine
    voice_engine.shutdown()
    voice_engine = VoiceEngine()
    if path is None:
        path, kind = _open_file()
        if not path: return
        if kind == "video":
            analyze_video(path); return
    frame  = cv2.imread(path)
    output = process_frame(frame, 0, 0, is_image=True)
    cv2.imshow("VIDAS  ·  IMAGE ANALYSIS", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────
# ANIMATED BACKGROUND CANVAS
# ──────────────────────────────────────────────────────────
class TechBackground(tk.Canvas):
    CHARS = "01アイウエオカキクケコ▣▤▥◈◉⬡△▽◁▷"

    def __init__(self, master, **kw):
        super().__init__(master, bg="#0a0a0a",
                         highlightthickness=0, **kw)
        self._particles = []
        self._t         = 0
        self._after_id  = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, e):
        self._W = e.width
        self._H = e.height
        if not self._particles:
            self._init_particles()
        self._tick()

    def _init_particles(self):
        self._particles = []
        for _ in range(38):
            self._particles.append({
                "x":  random.uniform(0, self._W),
                "y":  random.uniform(0, self._H),
                "vx": random.uniform(-0.3, 0.3),
                "vy": random.uniform(-0.4, -0.1),
                "r":  random.uniform(1.2, 3.0),
                "a":  random.uniform(0, math.pi * 2),
            })

    def _tick(self):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._draw()
        self._after_id = self.after(42, self._tick)

    def _draw(self):
        self.delete("all")
        W, H = self._W, self._H
        self._t += 1
        horizon = int(H * 0.52)
        vp_x    = W // 2
        n_lines = 14
        spread  = W * 1.1
        for i in range(n_lines + 1):
            bx   = int(-spread/2 + i*spread/n_lines)
            fade = abs(i - n_lines/2) / (n_lines/2)
            grey = int(18 + fade*10)
            self.create_line(vp_x, horizon, bx, H+40,
                             fill=f"#{grey:02x}{grey:02x}{grey:02x}", width=1)
        for j in range(1, 11):
            t    = j / 10
            y    = int(horizon + (H - horizon) * (t**1.8))
            grey = int(14 + t*18)
            self.create_line(0, y, W, y,
                             fill=f"#{grey:02x}{grey:02x}{grey:02x}", width=1)
        self.create_line(0, horizon, W, horizon, fill="#2a1800", width=2)
        for p in self._particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["a"] += 0.02
            if p["y"] < -10: p["y"] = H+5; p["x"] = random.uniform(0, W)
            if p["x"] < -10: p["x"] = W+5
            if p["x"] > W+10: p["x"] = -5
            r = p["r"] + math.sin(p["a"]) * 0.6
            x, y = int(p["x"]), int(p["y"])
            self.create_oval(x-r*3, y-r*3, x+r*3, y+r*3,
                             outline="#1a0e00", width=1)
            self.create_oval(x-r, y-r, x+r, y+r, fill="#FFB300", outline="")
        m, ln = 18, 28
        for (cx, cy, dx, dy) in [
            (m, m, 1, 1), (W-m, m, -1, 1),
            (m, H-m, 1, -1), (W-m, H-m, -1, -1),
        ]:
            self.create_line(cx, cy, cx+dx*ln, cy, fill="#FFB300", width=2)
            self.create_line(cx, cy, cx, cy+dy*ln, fill="#FFB300", width=2)


# ──────────────────────────────────────────────────────────
# GUI HELPERS
# ──────────────────────────────────────────────────────────
def _pulse_dot(canvas, oid, on_col, step=0):
    col = on_col if step % 2 == 0 else "#111111"
    canvas.itemconfig(oid, fill=col, outline=col)
    canvas.after(900, _pulse_dot, canvas, oid, on_col, step+1)


def _status_chip(parent, text, dot_color):
    f = ctk.CTkFrame(parent, fg_color="#181818",
                     corner_radius=20, border_width=1, border_color="#2e2e2e")
    f.pack(side="left", padx=5)
    c = ctk.CTkCanvas(f, width=10, height=10,
                      bg="#181818", highlightthickness=0)
    c.pack(side="left", padx=(12, 4), pady=9)
    oid = c.create_oval(1, 1, 9, 9, fill=dot_color, outline=dot_color)
    _pulse_dot(c, oid, dot_color)
    ctk.CTkLabel(f, text=text, font=("Consolas", 11, "bold"),
                 text_color=WHITE).pack(side="left", padx=(0, 14), pady=9)


def _module_grid(parent):
    modules = [
        ("⬡  LANE DETECTION",     AMBER),
        ("⬡  OBJECT DETECTION",   AMBER),
        ("⬡  TRAFFIC SIGNALS",    AMBER),
        ("⬡  POTHOLE DETECTION",  AMBER),
        ("⬡  FOG ESTIMATION",     BLUE_ICE),
        ("⬡  NIGHT VISION",       BLUE_ICE),
        ("⬡  VOICE ALERTS",       BLUE_ICE),
        ("⬡  PRIORITY HIERARCHY", BLUE_ICE),
    ]
    grid = ctk.CTkFrame(parent, fg_color="transparent")
    grid.pack(pady=(0, 28))
    for i, (name, col) in enumerate(modules):
        row, c = divmod(i, 4)
        chip = ctk.CTkFrame(grid, fg_color="#131313",
                            corner_radius=8, border_width=1,
                            border_color="#2a2a2a",
                            width=148, height=38)
        chip.grid(row=row, column=c, padx=5, pady=4)
        chip.pack_propagate(False)
        ctk.CTkLabel(chip, text=name,
                     font=("Consolas", 10, "bold"),
                     text_color=col).pack(expand=True)


# ──────────────────────────────────────────────────────────
# MAIN GUI
# ──────────────────────────────────────────────────────────
def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    app = ctk.CTk()
    app.title("VIDAS  ·  Intelligent Driver Assistance System")
    app.configure(fg_color=BG)
    app.after(0, lambda: app.state("zoomed"))

    bg_canvas = TechBackground(app)
    bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    root = ctk.CTkFrame(app, fg_color="transparent")
    root.place(relx=0.5, rely=0.5, anchor="center")

    ctk.CTkFrame(root, fg_color=AMBER,
                 height=3, width=680, corner_radius=2).pack(pady=(0, 18))
    ctk.CTkLabel(root,
                 text="VISION-BASED INTELLIGENT DRIVER ASSISTANCE SYSTEM",
                 font=("Consolas", 25), text_color=GREY_LIGHT,
                 fg_color="transparent").pack()
    ctk.CTkLabel(root, text="VIDAS",
                 font=("Impact", 96), text_color=WHITE,
                 fg_color="transparent").pack(pady=(0, 0))
    ctk.CTkFrame(root, fg_color=AMBER,
                 height=3, width=680, corner_radius=2).pack(pady=(0, 22))

    #chips = ctk.CTkFrame(root, fg_color="transparent")
    #chips.pack(pady=(0, 22))
    #_status_chip(chips, "AI CORE ONLINE",   GREEN)
    #_status_chip(chips, "VOICE READY",      BLUE_ICE)
    #_status_chip(chips, "FOG LOGIC ACTIVE", AMBER)
    #_status_chip(chips, "YOLOv8 LOADED",    GREEN)

    ctk.CTkButton(
        root, text="▶    LIVE VIDEO ANALYSIS",
        command=analyze_video,
        width=640, height=72,
        font=("Consolas", 18, "bold"),
        fg_color="#161005", hover_color="#1e1600",
        text_color=AMBER, corner_radius=6,
        border_width=2, border_color=AMBER
    ).pack(pady=(0, 12))

    ctk.CTkButton(
        root, text="◈    ANALYZE IMAGE",
        command=analyze_image,
        width=640, height=72,
        font=("Consolas", 18, "bold"),
        fg_color="#09121a", hover_color="#0c1820",
        text_color=BLUE_ICE, corner_radius=6,
        border_width=2, border_color=BLUE_ICE
    ).pack(pady=(0, 22))

    _module_grid(root)

    ctk.CTkButton(
        root, text="✕   SHUTDOWN",
        command=app.destroy,
        width=180, height=40,
        font=("Consolas", 13, "bold"),
        fg_color="transparent", hover_color="#1a0000",
        text_color=RED, corner_radius=6,
        border_width=1, border_color=RED
    ).pack()

    ctk.CTkFrame(root, fg_color=AMBER,
                 height=3, width=680, corner_radius=2).pack(pady=(22, 0))

    app.mainloop()


if __name__ == "__main__":
    main()