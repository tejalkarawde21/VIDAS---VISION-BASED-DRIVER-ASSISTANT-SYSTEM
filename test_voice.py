from voice_engine import VoiceEngine
import time

voice = VoiceEngine()

voice.speak(["System voice test successful."], cooldown=0)

time.sleep(5)