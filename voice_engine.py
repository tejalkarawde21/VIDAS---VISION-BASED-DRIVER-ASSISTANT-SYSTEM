import pyttsx3
import threading
import queue
import time


class VoiceEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.repeat_interval = 3
        self.last_spoken_time = {}
        self.active_messages = set()
        self.running = True

        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    # ── background TTS worker ─────────────────────────────
    def _speech_worker(self):
        while self.running:
            try:
                message = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                engine = pyttsx3.init(driverName="sapi5")
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)

                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[0].id)

                engine.say(message)
                engine.runAndWait()
                engine.stop()

            except Exception:
                pass

            self.queue.task_done()

    # ── called from process_frame for every alert ─────────
    def speak(self, message: str):
        if not message:
            return

        self.active_messages.add(message)
        now = time.time()

        if message not in self.last_spoken_time:
            self.queue.put(message)
            self.last_spoken_time[message] = now
            return

        if now - self.last_spoken_time[message] >= self.repeat_interval:
            self.queue.put(message)
            self.last_spoken_time[message] = now

    # ── called once per frame AFTER all speak() calls ─────
    def update_cycle(self):
        inactive = set(self.last_spoken_time.keys()) - self.active_messages
        for msg in inactive:
            del self.last_spoken_time[msg]
        self.active_messages.clear()

    # ── clear state between sessions ──────────────────────
    def reset(self):
        self.last_spoken_time = {}
        self.active_messages.clear()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass

    # ── shutdown ──────────────────────────────────────────
    def shutdown(self):
        self.running = False
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass