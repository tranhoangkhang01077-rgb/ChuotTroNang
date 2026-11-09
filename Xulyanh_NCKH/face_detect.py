# -*- coding: utf-8 -*-
"""Gyroscope mouse + eye gestures + optional Vietnamese voice dictation.

Quick start
-----------
Install: pip install opencv-python mediapipe numpy pyautogui pillow pyserial
Optional voice: pip install SpeechRecognition PyAudio
Run: python face_detect.py

Serial contract (deliberately configurable; no MPU6050 protocol is assumed):
  * COM_PORT / SERIAL_BAUD choose the USB serial connection.
  * SERIAL_FORMAT is "csv3" (default: one newline-terminated `x,y,z` sample)
    or "csv2" (`x,y`). Values are gyro angular rates in degrees/second, in
    X/Y/Z order for csv3. Change GYRO_X_INDEX / GYRO_Y_INDEX if needed.
  * Stay still for SERIAL_NEUTRAL_SECONDS after connecting. If firmware emits
    another format or units, adapt parse_serial_line() and sensitivity values.
  * Pointer motion is proportional to rate x elapsed time; scroll mode uses
    GYRO_SCROLL_AXIS. Serial reads always run in a daemon worker.

On first launch, follow the eye-calibration prompts. The numeric eye thresholds
are saved locally and reused on later launches; press R to recalibrate.
F8 toggles dictation, F9 pauses/resumes eye gestures, ESC exits. These hotkeys
are global on Windows when RegisterHotKey is available; otherwise they work
while this window has focus. Voice uses SpeechRecognition's Google recognizer
(internet required); no offline recognition/model is bundled.
E toggles low-light image enhancement; R repeats open/closed-eye calibration.
"""

import csv
import ctypes
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import messagebox, ttk

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import pyautogui
    from PIL import Image, ImageTk
except Exception as exc:
    print("Required camera/UI packages are unavailable:", exc)
    raise


# --- Camera and MediaPipe ---
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
FRAME_W, FRAME_H = 1280, 720
MAX_INFERENCE_WIDTH = 1280
TARGET_FRAME_INTERVAL_MS = 40
SMALL_EYE_WIDTH_PIXELS = 24
MIN_TRACKABLE_EYE_WIDTH_PIXELS = 12
MIN_EYE_LANDMARK_VISIBILITY = 0.5
MODEL_FILENAME = "face_landmarker.task"
CALIBRATION_FILE = os.path.join(
    os.getenv("LOCALAPPDATA", os.path.expanduser("~")),
    "Xulyanh_NCKH", "eye_calibration.json")
EYE_FEEDBACK_FILE = os.path.join(
    os.getenv("LOCALAPPDATA", os.path.expanduser("~")),
    "Xulyanh_NCKH", "eye_feedback.jsonl")
DETECT_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
ENHANCE_ENABLED = True
ENHANCE_TARGET_BRIGHTNESS = 115.0
ENHANCE_MIN_GAMMA = 0.45
CALIBRATION_OPEN_SECONDS = 2.0
CALIBRATION_BLINK_TIMEOUT_SECONDS = 20.0
CALIBRATION_MIN_CLOSED_SAMPLES = 2
CALIBRATION_CLOSED_RATIO = 0.82
CALIBRATION_OTHER_EYE_OPEN_RATIO = 0.82
CALIBRATION_BLINK_CLOSED_DELTA = 0.10
CALIBRATION_BLINK_OTHER_EYE_DELTA = 0.15
AUTO_BASELINE_MIN_SAMPLES = 30
AUTO_BASELINE_TIMEOUT_SECONDS = 20.0
AUTO_BASELINE_CLOSED_SCORE_MAX = 0.45
AUTO_BASELINE_EAR_THRESHOLD_RATIO = 0.90
AUTO_BASELINE_BLINK_MARGIN = 0.12
AUTO_BASELINE_ADAPT_RATE = 0.001
EYE_DEFAULT_OPEN_EAR = 0.28
EYE_DEFAULT_THRESHOLD = 0.19
EYE_HYSTERESIS = 0.018
BLENDSHAPE_STATE_MARGIN = 0.03
EYE_FEEDBACK_WINDOW_SECONDS = 3.0
EYE_FEEDBACK_EAR_STEP = 0.005
EYE_FEEDBACK_BLENDSHAPE_STEP = 0.03
EYE_FEEDBACK_HISTORY_SIZE = 300

# Eye contours. The stream is mirrored before detection; classify "left" and
# "right" from each contour's position in that mirrored image so the displayed
# viewer-left eye (the user's left eye in the mirror) is consistently "LEFT".
EYE_CONTOURS = (
    ((33, 133), ((160, 144), (159, 145), (158, 153))),
    ((362, 263), ((385, 380), (386, 374), (387, 373))),
)
EYE_DRAW_POINTS = (
    (33, 160, 159, 158, 133, 153, 145, 144),
    (362, 385, 386, 387, 263, 373, 374, 380),
)
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)

# --- Gyro serial settings; edit here or set matching environment variables ---
COM_PORT = os.getenv("COM_PORT", "COM4")
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
SERIAL_FORMAT = os.getenv("SERIAL_FORMAT", "csv3").lower()
GYRO_X_INDEX = int(os.getenv("GYRO_X_INDEX", "0"))
GYRO_Y_INDEX = int(os.getenv("GYRO_Y_INDEX", "1"))
GYRO_SCROLL_AXIS = os.getenv("GYRO_SCROLL_AXIS", "y").lower()
SERIAL_NEUTRAL_SECONDS = 2.5
GYRO_STALE_SECONDS = 0.25
GYRO_DEADZONE_DPS = 2.0
GYRO_POINTER_SENSITIVITY = 7.0       # pixels per (degree/second * second)
GYRO_SCROLL_SENSITIVITY = 0.035      # wheel steps per (degree/second * second)
POINTER_SMOOTHING = 0.28

# --- Eye gesture timing ---
QUICK_BLINK_MIN_SECONDS = 0.08
QUICK_BLINK_MAX_SECONDS = 1.4
BLINK_SEQUENCE_GAP_SECONDS = 0.75
DRAG_HOLD_SECONDS = 1.65
SCROLL_HOLD_SECONDS = 1.65
DOUBLE_CLOSED_EXIT_SECONDS = 3.0
EYE_STATE_DEBOUNCE_SECONDS = 0.06
SCROLL_AXIS_SIGN = 1
VOICE_AMBIENT_CALIBRATION_SECONDS = 0.8
VOICE_START_TIMEOUT_SECONDS = 5.0
VOICE_PHRASE_TIME_LIMIT_SECONDS = 15.0
VOICE_PAUSE_THRESHOLD_SECONDS = 1.0

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


def clamp(value, low, high):
    return max(low, min(high, value))


def eye_aspect_ratio(points, contour):
    """Return normalized eye aperture; values are invariant to face scale."""
    corners, pairs = contour
    horizontal = float(np.linalg.norm(points[corners[0]] - points[corners[1]]))
    if horizontal < 1e-6:
        return None
    vertical = sum(float(np.linalg.norm(points[a] - points[b])) for a, b in pairs)
    return (vertical / len(pairs)) / horizontal


def parse_serial_line(line, data_format=SERIAL_FORMAT):
    """Parse a documented CSV sample; reject malformed, non-finite values."""
    if data_format not in ("csv2", "csv3"):
        raise ValueError("SERIAL_FORMAT must be csv2 or csv3")
    try:
        values = [float(value.strip()) for value in next(csv.reader([line]))]
    except (ValueError, StopIteration, csv.Error) as exc:
        raise ValueError(f"invalid {data_format} sample: {line!r}") from exc
    expected = 2 if data_format == "csv2" else 3
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected {expected} finite comma-separated values")
    return tuple(values)


def select_google_transcript(response):
    """Choose the highest-confidence transcript from Google's alternatives."""
    if isinstance(response, str):
        return response.strip()
    if not isinstance(response, dict):
        return ""
    alternatives = [
        alternative for alternative in response.get("alternative", [])
        if isinstance(alternative, dict)
        and isinstance(alternative.get("transcript"), str)
        and alternative["transcript"].strip()
    ]
    if not alternatives:
        return ""
    def confidence(alternative):
        try:
            value = float(alternative.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    best = max(alternatives, key=confidence)
    return best["transcript"].strip()


def audio_rms_level(audio_data):
    """Return captured audio RMS without saving or exposing the recording."""
    raw_audio = audio_data.get_raw_data(convert_rate=16_000, convert_width=2)
    if not raw_audio:
        return 0.0
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def list_microphone_devices():
    """Return input devices with their native rates and the system default."""
    try:
        import pyaudio
    except ImportError:
        default = ("System default microphone", None, None)
        return [default], default

    audio = pyaudio.PyAudio()
    try:
        default_index = int(audio.get_default_input_device_info()["index"])
        default_info = audio.get_device_info_by_index(default_index)
        default_rate = int(default_info["defaultSampleRate"])
        default_name = " ".join(str(default_info.get("name", "")).split())
        default_label = (
            f"System default ({default_rate / 1000:g} kHz) - {default_name}")
        devices = [(default_label, None, default_rate)]
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                name = " ".join(str(info.get("name", f"Input {index}")).split())
                rate = int(info["defaultSampleRate"])
                rate_note = " LOW-BAND" if rate < 16_000 else ""
                devices.append(
                    (f"{index}: {rate / 1000:g} kHz{rate_note} - {name}",
                     index, rate))
        default_device = next(
            (device for device in devices if device[1] == default_index),
            devices[0])
        return devices, default_device
    finally:
        audio.terminate()


class LowLightEnhancer:
    """Apply an inexpensive adaptive gamma LUT to model input only."""

    def __init__(self):
        self.enabled = ENHANCE_ENABLED

    def process(self, frame):
        brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if not self.enabled or brightness >= ENHANCE_TARGET_BRIGHTNESS:
            return frame
        normalized = clamp(brightness / 255.0, 1e-3, 1.0)
        target = ENHANCE_TARGET_BRIGHTNESS / 255.0
        gamma = clamp(math.log(target) / math.log(normalized),
                      ENHANCE_MIN_GAMMA, 1.0)
        lut = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255,
            0, 255).astype(np.uint8)
        return cv2.LUT(frame, lut)


@dataclass
class EyeCalibration:
    open_left: float = EYE_DEFAULT_OPEN_EAR
    open_right: float = EYE_DEFAULT_OPEN_EAR
    threshold_left: float = EYE_DEFAULT_THRESHOLD
    threshold_right: float = EYE_DEFAULT_THRESHOLD
    blink_threshold_left: float | None = None
    blink_threshold_right: float | None = None
    ready: bool = False


class GestureController:
    """Debounced eye gesture FSM; callbacks are the only mouse action points."""

    def __init__(self, action, set_scroll):
        self.action = action
        self.set_scroll = set_scroll
        self.scroll_mode = False
        self.reset()

    def reset(self):
        self.previous = {"left": None, "right": None}
        self.candidate = {"left": None, "right": None}
        self.candidate_since = {"left": None, "right": None}
        self.closed_at = {"left": None, "right": None}
        self.pending = {"left": None, "right": None}
        self.both_closed_at = None
        self.dragging = False
        self.drag_candidate = False
        self.drag_button_down = False

    def _click(self, button):
        if not self.scroll_mode:
            self.action("click", button)

    def _queue_blink(self, eye, count, now):
        self.pending[eye] = {"count": count, "last": now}

    def _end_drag(self):
        if self.drag_button_down:
            self.action("up", "left")
        self.dragging = False
        self.drag_button_down = False
        self.drag_candidate = False

    def lost_face(self):
        self._end_drag()
        self.reset()
        if self.scroll_mode:
            self.scroll_mode = False
            self.set_scroll(False)

    def _end_scroll(self):
        if self.scroll_mode:
            self.scroll_mode = False
            self.set_scroll(False)

    def update(self, now, left, right):
        observed = {"left": left, "right": right}
        if None in self.previous.values():
            self.previous = observed.copy()
            return

        prior_states = self.previous.copy()
        was_holding = self.dragging or self.scroll_mode
        hold_eye = "left" if self.dragging else (
            "right" if self.scroll_mode else None)
        stable_states = prior_states.copy()
        for eye, state in observed.items():
            if state == prior_states[eye]:
                self.candidate[eye] = None
                self.candidate_since[eye] = None
            elif self.candidate[eye] != state:
                self.candidate[eye] = state
                self.candidate_since[eye] = now
            elif (self.candidate_since[eye] is not None
                  and now - self.candidate_since[eye] >= EYE_STATE_DEBOUNCE_SECONDS):
                stable_states[eye] = state
                self.candidate[eye] = None
                self.candidate_since[eye] = None

        states = stable_states
        was_both_closed = (
            prior_states["left"] == "CLOSED"
            and prior_states["right"] == "CLOSED")
        opened = []
        for eye, state in states.items():
            if state == "CLOSED" and prior_states[eye] == "OPEN":
                self.closed_at[eye] = now
            elif state == "OPEN" and prior_states[eye] == "CLOSED":
                started = self.closed_at[eye]
                duration = now - started if started is not None else 0
                self.closed_at[eye] = None
                opened.append((eye, duration))

        both_closed = (
            states["left"] == "CLOSED" and states["right"] == "CLOSED")
        if both_closed:
            if self.both_closed_at is None:
                self.both_closed_at = now
            elif now - self.both_closed_at >= DOUBLE_CLOSED_EXIT_SECONDS:
                self._end_drag()
                self._end_scroll()
                self.pending["left"] = None
                self.pending["right"] = None
        else:
            self.both_closed_at = None

        if both_closed and not was_holding:
            self.closed_at["left"] = None
            self.closed_at["right"] = None
            self.pending["left"] = None

        if (states["left"] == "CLOSED" and states["right"] == "OPEN"
                and self.closed_at["left"] is not None
                and not self.scroll_mode and not self.dragging
                and now - self.closed_at["left"] >= DRAG_HOLD_SECONDS):
            self.pending["left"] = None
            self.dragging = True
            self.action("down", "left")
            self.drag_button_down = True

        if (states["right"] == "CLOSED" and states["left"] == "OPEN"
                and self.closed_at["right"] is not None
                and not self.scroll_mode
                and now - self.closed_at["right"] >= SCROLL_HOLD_SECONDS):
            self.pending["left"] = None
            self.scroll_mode = True
            self.set_scroll(True)

        for eye, duration in opened:
            if eye == "left" and self.dragging:
                self._end_drag()
            if eye == "right" and self.scroll_mode:
                self._end_scroll()
            if was_both_closed:
                continue
            if was_holding and eye != hold_eye:
                continue
            self._opened(eye, duration, now)

        self.previous = states.copy()

    def _opened(self, eye, duration, now):
        if self.scroll_mode:
            return
        if not QUICK_BLINK_MIN_SECONDS <= duration <= QUICK_BLINK_MAX_SECONDS:
            self.pending[eye] = None
            return

        if eye == "right":
            self._click("right")
            return

        pending = self.pending[eye]
        if pending is not None and now - pending["last"] <= BLINK_SEQUENCE_GAP_SECONDS:
            if pending["count"] == 1:
                pending["count"] = 2
                pending["last"] = now
            else:
                self.action("multi_click", 2)
                self._queue_blink(eye, 1, now)
        else:
            self._queue_blink(eye, 1, now)

        self.pending["right"] = None

    def flush(self, now):
        if self.scroll_mode:
            self.pending["left"] = None
            return
        pending = self.pending["left"]
        if pending and now - pending["last"] > BLINK_SEQUENCE_GAP_SECONDS:
            self.action("multi_click", pending["count"])
            self.pending["left"] = None


class SerialGyro(threading.Thread):
    """Owns serial I/O and publishes the latest calibrated sample."""

    def __init__(self, publish_status):
        super().__init__(name="gyro-serial", daemon=True)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.status = "starting"
        self.status_callback = publish_status
        self.neutral = None
        self.samples = []
        self.first_sample = None

    def set_status(self, text):
        self.status = text
        self.status_callback(text)

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        try:
            import serial
        except ImportError:
            self.set_status("OFF — install pyserial")
            return
        if SERIAL_FORMAT not in ("csv2", "csv3"):
            self.set_status("ERROR — SERIAL_FORMAT must be csv2/csv3")
            return
        try:
            with serial.Serial(COM_PORT, SERIAL_BAUD, timeout=0.1) as port:
                self.set_status(f"connected {COM_PORT}; keep gyro still to calibrate")
                while not self.stop_event.is_set():
                    raw = port.readline()
                    if not raw:
                        continue
                    try:
                        values = parse_serial_line(raw.decode("ascii", errors="strict").strip())
                        x, y = values[GYRO_X_INDEX], values[GYRO_Y_INDEX]
                    except (UnicodeError, ValueError, IndexError) as exc:
                        self.set_status(f"ERROR — serial data: {exc}")
                        continue
                    now = time.monotonic()
                    if self.first_sample is None:
                        self.first_sample = now
                    if self.neutral is None:
                        self.samples.append((x, y))
                        if now - self.first_sample >= SERIAL_NEUTRAL_SECONDS:
                            self.neutral = (
                                sum(v[0] for v in self.samples) / len(self.samples),
                                sum(v[1] for v in self.samples) / len(self.samples),
                            )
                            self.set_status("connected; neutral calibrated")
                    with self.lock:
                        self.latest = (x, y, now, self.neutral)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.set_status(f"ERROR — {COM_PORT}: {exc}")

    def stop(self):
        self.stop_event.set()


class VoiceDictation(threading.Thread):
    """Optional asynchronous Google SpeechRecognition client, Vietnamese only."""

    def __init__(self, post_status, post_text):
        super().__init__(name="voice-dictation", daemon=True)
        self.active = threading.Event()
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.generation = 0
        self.microphone_device_index = None
        self.microphone_sample_rate = None
        self.post_status = post_status
        self.post_text = post_text

    def current_generation(self):
        with self.state_lock:
            return self.generation

    def is_active_session(self, generation):
        with self.state_lock:
            return (self.generation == generation and self.active.is_set()
                    and not self.stop_event.is_set())

    def activate(self):
        with self.state_lock:
            if self.stop_event.is_set():
                return self.generation
            if not self.active.is_set():
                self.generation += 1
                self.active.set()
            return self.generation

    def select_microphone(self, device_index, sample_rate=None):
        with self.state_lock:
            if (device_index == self.microphone_device_index
                    and sample_rate == self.microphone_sample_rate):
                return
            self.microphone_device_index = device_index
            self.microphone_sample_rate = sample_rate
            if self.active.is_set() and not self.stop_event.is_set():
                self.generation += 1

    def _selected_microphone(self):
        with self.state_lock:
            return self.microphone_device_index, self.microphone_sample_rate

    def _fail_session(self, generation):
        with self.state_lock:
            if self.generation != generation or self.stop_event.is_set():
                return False
            self.active.clear()
            return True

    def _active_generation(self):
        with self.state_lock:
            if self.active.is_set() and not self.stop_event.is_set():
                return self.generation
        return None

    def run(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self.post_status(
                "ERROR — install SpeechRecognition and PyAudio",
                self.current_generation())
            return
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 5
        recognizer.pause_threshold = VOICE_PAUSE_THRESHOLD_SECONDS
        calibrated_generation = None
        while not self.stop_event.is_set():
            if not self.active.wait(0.15):
                continue
            generation = self._active_generation()
            if generation is None:
                continue
            try:
                device_index, sample_rate = self._selected_microphone()
                with sr.Microphone(
                        device_index=device_index,
                        sample_rate=sample_rate) as microphone:
                    if not self.is_active_session(generation):
                        continue
                    if calibrated_generation != generation:
                        self.post_status("CALIBRATING_MIC", generation)
                        recognizer.adjust_for_ambient_noise(
                            microphone, duration=VOICE_AMBIENT_CALIBRATION_SECONDS)
                        calibrated_generation = generation
                    if not self.is_active_session(generation):
                        continue
                    self.post_status("LISTENING", generation)
                    audio = recognizer.listen(
                        microphone, timeout=VOICE_START_TIMEOUT_SECONDS,
                        phrase_time_limit=VOICE_PHRASE_TIME_LIMIT_SECONDS)
                if not self.is_active_session(generation):
                    continue
                self.post_status(
                    f"AUDIO_LEVEL:{audio_rms_level(audio):.0f}", generation)
                self.post_status("PROCESSING", generation)
                response = recognizer.recognize_google(
                    audio, language="vi-VN", show_all=True)
                text = select_google_transcript(response)
                if self.is_active_session(generation) and text.strip():
                    self.post_text(text.strip(), generation)
                elif self.is_active_session(generation):
                    self.post_status("UNRECOGNIZED", generation)
                if self.is_active_session(generation):
                    self.post_status("LISTENING", generation)
            except sr.WaitTimeoutError:
                if self.is_active_session(generation):
                    self.post_status("NO_SPEECH", generation)
                continue
            except sr.UnknownValueError:
                if self.is_active_session(generation):
                    self.post_status("UNRECOGNIZED", generation)
            except sr.RequestError as exc:
                if not self._fail_session(generation):
                    continue
                self.post_status(f"ERROR — online recognition: {exc}", generation)
            except Exception as exc:
                if not self._fail_session(generation):
                    continue
                self.post_status(f"ERROR — microphone: {exc}", generation)

    def deactivate(self):
        with self.state_lock:
            if self.active.is_set():
                self.active.clear()
                self.generation += 1

    def stop(self):
        self.deactivate()
        with self.state_lock:
            self.stop_event.set()


class FaceGyroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gyro Mouse · Eye Gestures · Vietnamese Dictation")
        screen_width, screen_height = pyautogui.size()
        window_width = min(1600, screen_width)
        window_height = min(900, screen_height)
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min(1100, screen_width), min(650, screen_height))
        self.running = True
        self.paused = False
        self.scroll_mode = False
        self.gestures_armed = False
        self.face_detected = False
        self.eye_tracking_status = "WAITING FOR CAMERA"
        self.voice_status = "OFF"
        self.voice_feedback = "Chưa có câu nhận dạng. Nói xong chờ một chút để xem kết quả."
        self.gyro_status = "starting"
        self.last_command = "none"
        self.error_status = ""
        self.last_frame_time = time.monotonic()
        self.fps = 0.0
        self.calibration_started = time.monotonic()
        self.calibration_stage_started = self.calibration_started
        self.calibration_values = {
            "open": [[], []], "closed": [[], []],
            "blink_open": [[], []], "blink_closed": [[], []],
        }
        self.calibration_closed_samples = 0
        self.calibration = self._load_eye_calibration()
        self.calibration_stage = "auto"
        self.auto_baseline_started = self.calibration_started
        self.auto_baseline_samples = {"ear": [[], []], "blink": [[], []]}
        self.enhancer = LowLightEnhancer()
        self.eye_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
        self.last_ears = (None, None)
        self.last_eye_widths = (None, None)
        self.camera_resolution = (0, 0)
        self.last_blink_scores = (None, None)
        self.eye_feature_history = deque(maxlen=EYE_FEEDBACK_HISTORY_SIZE)
        self.eye_feedback_count = 0
        self.eye_feedback_status = "Chưa ghi phản hồi"
        self.last_mode = "NORMAL"
        try:
            self.microphone_devices, default_microphone = list_microphone_devices()
            self.microphone_error = ""
        except (OSError, ValueError) as exc:
            default_microphone = ("System default microphone", None, None)
            self.microphone_devices = [default_microphone]
            self.microphone_error = f"Could not list microphones: {exc}"
        self.default_microphone = default_microphone
        self.pointer = pyautogui.position()
        self.filtered_pointer = [float(self.pointer.x), float(self.pointer.y)]
        self.last_gyro_tick = time.monotonic()
        self.scroll_fraction = 0.0
        self.last_mouse_button = None
        self.image_ref = None
        self.callback_queue = queue.Queue()
        self._registered_hotkeys = False
        self._msg = None

        self._build_ui()
        self.cap = self._open_camera()
        try:
            self.landmarker = self._create_landmarker()
        except Exception:
            self.cap.release()
            raise
        self.gyro = SerialGyro(
            lambda text: self._post(lambda value=text: self._set_gyro_status(value)))
        self.voice = VoiceDictation(
            lambda text, generation: self._post(
                lambda value=text, session=generation:
                self._set_voice_status(value, session)),
            lambda text, generation: self._post(
                lambda value=text, session=generation:
                self._insert_text(value, session)))
        selected_device = self.microphone_device_choices.get(
            self.microphone_choice.get())
        selected_rate = self.microphone_sample_rates.get(self.microphone_choice.get())
        self.voice.select_microphone(selected_device, selected_rate)
        self.gyro.start()
        self.voice.start()
        self.gestures = GestureController(self._mouse_action, self._set_scroll_mode)
        self._register_hotkeys()
        self.root.bind_all("<Escape>", lambda _event: self.shutdown())
        if not self._registered_hotkeys:
            self.root.bind_all("<F8>", lambda _event: self.toggle_voice())
            self.root.bind_all("<F9>", lambda _event: self.toggle_pause())
        self.root.bind_all(
            "<KeyPress-r>", lambda event: self._handle_shortcut(
                event, self.start_eye_calibration))
        self.root.bind_all(
            "<KeyPress-R>", lambda event: self._handle_shortcut(
                event, self.start_eye_calibration))
        self.root.bind_all(
            "<KeyPress-e>", lambda event: self._handle_shortcut(
                event, self.toggle_enhancement))
        self.root.bind_all(
            "<KeyPress-E>", lambda event: self._handle_shortcut(
                event, self.toggle_enhancement))
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self._tick()
        self._update_status()
        print("Keep the gyro still during neutral calibration. F8 voice, F9 pause, R recalibrate, E enhance, ESC exit.")
        print("Voice dictation uses Google's online Vietnamese recognizer (internet required).")

    def _build_ui(self):
        self.root.configure(bg="#20252b")
        header = tk.Frame(self.root, bg="#20252b")
        header.pack(fill="x", padx=16, pady=10)
        tk.Label(header, text="GYRO MOUSE  ·  EYE GESTURES  ·  VOICE",
                 font=("Segoe UI", 17, "bold"), fg="white", bg="#20252b").pack(side="left")
        self.mode_label = tk.Label(header, text="NORMAL", font=("Segoe UI", 14, "bold"),
                                   fg="#75e0a1", bg="#20252b")
        self.mode_label.pack(side="right", padx=12)

        controls = tk.Frame(self.root, bg="#303840")
        controls.pack(fill="x", padx=16)
        for text, callback in (
            ("F8 · Voice dictation", self.toggle_voice),
            ("F9 · Pause eye gestures", self.toggle_pause),
            ("R · Recalibrate eyes", self.start_eye_calibration),
            ("E · Light enhancement", self.toggle_enhancement),
            ("ESC · Exit", self.shutdown),
        ):
            ttk.Button(controls, text=text, command=callback).pack(side="left", padx=6, pady=7)

        body = tk.Frame(self.root, bg="#20252b")
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.grid_columnconfigure(0, weight=3, uniform="panels")
        body.grid_columnconfigure(1, weight=2, uniform="panels")
        body.grid_rowconfigure(0, weight=1)

        left_panel = tk.Frame(body, bg="#303840")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        self.preview = tk.Label(left_panel, bg="black", text="Opening camera…",
                                fg="white", font=("Segoe UI", 14))
        self.preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.status_text = tk.StringVar()
        tk.Label(left_panel, textvariable=self.status_text, bg="#303840", fg="white",
                 justify="left", anchor="nw", font=("Consolas", 10),
                 wraplength=850).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
        feedback = tk.LabelFrame(
            left_panel, text="Phản hồi để cải thiện nhận diện (ghi số liệu, không lưu ảnh)",
            bg="#303840", fg="white", padx=8, pady=5)
        feedback.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        feedback_labels = (
            ("Đúng", "success"),
            ("Bỏ sót nháy", "missed"),
            ("Nhận nhầm", "false_positive"),
        )
        for row, eye in enumerate(("left", "right")):
            tk.Label(feedback, text="MẮT TRÁI" if eye == "left" else "MẮT PHẢI",
                     bg="#303840", fg="white").grid(
                         row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            for column, (label, result) in enumerate(feedback_labels, start=1):
                ttk.Button(
                    feedback, text=label,
                    command=lambda side=eye, outcome=result:
                    self.record_eye_feedback(side, outcome)).grid(
                        row=row, column=column, padx=3, pady=2, sticky="ew")
        tk.Label(
            feedback,
            text="Đánh dấu trong vòng 3 giây sau lần thử; dùng F9 để tạm dừng thao tác mắt khi cần.",
            bg="#303840", fg="#d0d8e0", anchor="w").grid(
                row=2, column=0, columnspan=4, sticky="ew", pady=(3, 0))
        for column in range(1, 4):
            feedback.grid_columnconfigure(column, weight=1)

        voice_panel = tk.Frame(body, bg="#303840")
        voice_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        voice_panel.grid_rowconfigure(8, weight=1)
        voice_panel.grid_columnconfigure(0, weight=1)
        tk.Label(voice_panel, text="MÔ PHỎNG ĐỘ MỞ MẮT",
                 bg="#303840", fg="white", anchor="w",
                 font=("Segoe UI", 15, "bold")).grid(
                     row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.eye_simulation = tk.Canvas(
            voice_panel, width=360, height=150, bg="#20252b",
            highlightthickness=1, highlightbackground="#52606d")
        self.eye_simulation.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.calibration_guide_text = tk.StringVar()
        tk.Label(
            voice_panel, textvariable=self.calibration_guide_text,
            bg="#303840", fg="#d0d8e0", anchor="w", justify="left",
            wraplength=500, font=("Segoe UI", 10)).grid(
                row=2, column=0, sticky="ew", padx=14, pady=(2, 4))
        ttk.Button(
            voice_panel, text="Bắt đầu hướng dẫn / hiệu chuẩn lại (R)",
            command=self.start_eye_calibration).grid(
                row=3, column=0, sticky="ew", padx=14, pady=(2, 8))
        tk.Label(voice_panel, text="VOICE & TEXT TEST",
                 bg="#303840", fg="white", anchor="w",
                 font=("Segoe UI", 16, "bold")).grid(
                     row=4, column=0, sticky="ew", padx=14, pady=(14, 6))
        default_label = self.default_microphone[0]
        self.voice_panel_status = tk.StringVar(
            value="Voice: OFF · nhận dạng tiếng Việt (cần Internet)\n"
                  f"Mic: {default_label}\n"
                  "Chưa có câu nhận dạng. Nói xong chờ một chút để xem kết quả.")
        tk.Label(voice_panel, textvariable=self.voice_panel_status,
                 bg="#303840", fg="#d0d8e0", anchor="w",
                 font=("Segoe UI", 10)).grid(
                     row=5, column=0, sticky="ew", padx=14, pady=(0, 8))
        voice_controls = tk.Frame(voice_panel, bg="#303840")
        voice_controls.grid(row=6, column=0, sticky="ew", padx=10, pady=4)
        tk.Label(voice_controls, text="Micro:", bg="#303840", fg="white").pack(
            side="left", padx=(4, 2))
        self.microphone_device_choices = {
            label: index for label, index, _rate in self.microphone_devices}
        self.microphone_sample_rates = {
            label: rate for label, _index, rate in self.microphone_devices}
        microphone_labels = list(self.microphone_device_choices)
        self.microphone_choice = tk.StringVar(value=default_label)
        self.microphone_selector = ttk.Combobox(
            voice_controls, textvariable=self.microphone_choice,
            values=microphone_labels, state="readonly", width=32)
        self.microphone_selector.pack(side="left", padx=4, fill="x", expand=True)
        self.microphone_selector.bind(
            "<<ComboboxSelected>>", self._on_microphone_change)
        self.voice_button = ttk.Button(
            voice_controls, text="Start voice (F8)", command=self.toggle_voice)
        self.voice_button.pack(side="left", padx=4)
        ttk.Button(voice_controls, text="Clear", command=self._clear_voice_text).pack(
            side="left", padx=4)
        ttk.Button(voice_controls, text="Copy", command=self._copy_voice_text).pack(
            side="left", padx=4)

        self.voice_to_panel = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            voice_panel, text="Send recognized speech to this test area",
            variable=self.voice_to_panel).grid(
                row=7, column=0, sticky="nw", padx=14, pady=(8, 4))
        text_frame = tk.Frame(voice_panel, bg="#20252b")
        text_frame.grid(row=8, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self.voice_text = tk.Text(
            text_frame, wrap="word", undo=True, font=("Segoe UI", 12),
            bg="#f5f5f5", fg="#20252b", insertbackground="#20252b")
        text_scroll = ttk.Scrollbar(text_frame, orient="vertical",
                                    command=self.voice_text.yview)
        self.voice_text.configure(yscrollcommand=text_scroll.set)
        self.voice_text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")
        tk.Label(self.root,
                 text=f"Mirrored preview · LEFT means viewer-left in the mirror · "
                      f"red outline = detected face; green points = eye landmarks · "
                      f"blink/reopen ≤{QUICK_BLINK_MAX_SECONDS:.1f} s: click; "
                      f"hold ≥{DRAG_HOLD_SECONDS:.2f} s: drag/scroll "
                      f"(other eye may blink; open held eye to exit; "
                      f"both closed {DOUBLE_CLOSED_EXIT_SECONDS:.0f}s to exit)",
                 fg="#d0d8e0", bg="#20252b", font=("Segoe UI", 9),
                 justify="left", wraplength=1400).pack(
                     fill="x", padx=16, pady=(0, 8))

    def _clear_voice_text(self):
        self.voice_text.delete("1.0", "end")

    def _handle_shortcut(self, event, callback):
        if getattr(self, "voice_text", None) is not None \
                and event.widget is self.voice_text:
            return
        callback()

    def _copy_voice_text(self):
        text = self.voice_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _on_microphone_change(self, _event=None):
        if not hasattr(self, "voice"):
            return
        label = self.microphone_choice.get()
        self.voice.select_microphone(
            self.microphone_device_choices[label],
            self.microphone_sample_rates[label])
        sample_rate = self.microphone_sample_rates[label]
        if sample_rate is not None and sample_rate < 16_000:
            self.voice_feedback = (
                f"Mic này chỉ {sample_rate // 1000} kHz, dễ nhận sai tiếng Việt. "
                "Hãy chọn mic 16 kHz trở lên.")
        else:
            self.voice_feedback = (
                f"Đang chọn mic {sample_rate or 'mặc định'} Hz. "
                "Nói thử một câu trọn vẹn.")
        self.voice_panel_status.set(
            f"Voice: {self.voice_status} · nhận dạng tiếng Việt (cần Internet)\n"
            f"Mic: {label}\n{self.voice_feedback}")

    def _open_camera(self):
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {CAM_INDEX}; change CAM_INDEX.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        return cap

    def _create_landmarker(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
        if not os.path.isfile(model_path) or os.path.getsize(model_path) < 100_000:
            raise RuntimeError(f"MediaPipe model missing or invalid: {model_path}")
        try:
            vision = mp.tasks.vision
            options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=DETECT_CONFIDENCE,
                min_face_presence_confidence=DETECT_CONFIDENCE,
                min_tracking_confidence=TRACKING_CONFIDENCE,
                output_face_blendshapes=True,
            )
            self.landmarker_api = "tasks"
            return vision.FaceLandmarker.create_from_options(options)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"MediaPipe Tasks FaceLandmarker is not available in this version: {exc}. "
                "Install a MediaPipe release with mp.tasks.vision.FaceLandmarker."
            ) from exc

    def _post(self, callback):
        self.callback_queue.put(callback)

    def _set_gyro_status(self, text):
        self.gyro_status = text

    def _set_voice_status(self, text, generation):
        if generation != self.voice.current_generation():
            return
        if text.startswith("AUDIO_LEVEL:"):
            try:
                audio_level = float(text.partition(":")[2])
            except ValueError:
                audio_level = 0.0
            if audio_level < 180:
                self.voice_feedback = (
                    f"Âm thanh rất nhỏ (RMS {audio_level:.0f}); chọn đúng mic và nói gần hơn.")
            elif audio_level > 28000:
                self.voice_feedback = (
                    f"Âm thanh bị quá lớn/vỡ (RMS {audio_level:.0f}); giảm gain mic.")
            else:
                self.voice_feedback = f"Đã thu âm (RMS {audio_level:.0f}); đang gửi nhận dạng."
            text = "PROCESSING"
        elif text == "NO_SPEECH":
            self.voice_feedback = "Chưa phát hiện tiếng nói; kiểm tra micro và nói gần hơn."
            text = "LISTENING"
        elif text == "CALIBRATING_MIC":
            self.voice_feedback = "Đang đo tiếng ồn nền; chờ một chút rồi nói trọn câu tiếng Việt."
        elif text == "UNRECOGNIZED":
            self.voice_feedback = "Có âm thanh nhưng không hiểu được câu; thử nói chậm và rõ hơn."
            text = "LISTENING"
        if text in ("LISTENING", "PROCESSING") and not self.voice.is_active_session(generation):
            text = "OFF"
        self.voice_status = text
        self.voice_panel_status.set(
            f"Voice: {text} · nhận dạng tiếng Việt (cần Internet)\n"
            f"Mic: {self.microphone_choice.get()}\n{self.voice_feedback}")
        self.voice_button.configure(
            text="Stop voice (F8)" if self.voice.active.is_set() else "Start voice (F8)")

    def _load_eye_calibration(self):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as calibration_file:
                values = json.load(calibration_file)
        except FileNotFoundError:
            return EyeCalibration()
        except (OSError, json.JSONDecodeError) as exc:
            self.error_status = (
                f"Saved eye thresholds unavailable ({exc}); calibrate once or press R.")
            return EyeCalibration()

        try:
            if values.get("version") != 3:
                self.error_status = (
                    "Saved eye calibration is outdated; rebuilding it automatically. "
                    "Keep both eyes naturally open.")
                return EyeCalibration()
            open_left = float(values["open_left"])
            open_right = float(values["open_right"])
            threshold_left = float(values["threshold_left"])
            threshold_right = float(values["threshold_right"])
            thresholds = (open_left, open_right, threshold_left, threshold_right)
            if (not all(math.isfinite(value) for value in thresholds)
                    or not 0 < threshold_left < open_left < 1
                    or not 0 < threshold_right < open_right < 1):
                raise ValueError("invalid eye threshold values")
            if ("blink_threshold_left" not in values
                    or "blink_threshold_right" not in values):
                self.error_status = (
                    "Saved calibration predates blink scoring; complete this "
                    "one-time eye calibration.")
                return EyeCalibration()
            blink_threshold_left = values["blink_threshold_left"]
            blink_threshold_right = values["blink_threshold_right"]
            if blink_threshold_left is not None:
                blink_threshold_left = float(blink_threshold_left)
            if blink_threshold_right is not None:
                blink_threshold_right = float(blink_threshold_right)
            for threshold in (blink_threshold_left, blink_threshold_right):
                if threshold is not None and (
                        not math.isfinite(threshold) or not 0 < threshold < 1):
                    raise ValueError("invalid blink threshold values")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.error_status = (
                f"Saved eye thresholds are invalid ({exc}); calibrate once or press R.")
            return EyeCalibration()
        return EyeCalibration(
            open_left=open_left, open_right=open_right,
            threshold_left=threshold_left, threshold_right=threshold_right,
            blink_threshold_left=blink_threshold_left,
            blink_threshold_right=blink_threshold_right,
            ready=True)

    def _set_scroll_mode(self, enabled):
        self.scroll_mode = enabled
        self.scroll_fraction = 0.0

    def toggle_voice(self):
        if not self.running:
            return
        if self.voice.active.is_set():
            self.error_status = ""
            self.voice.deactivate()
            self.voice_status = "OFF"
        elif not self.voice.is_alive():
            self.voice_status = "ERROR — voice worker unavailable; restart the app"
        else:
            self.error_status = ""
            self.voice.activate()
            self.voice_status = "STARTING"
        self.voice_panel_status.set(
            f"Voice: {self.voice_status} · nhận dạng tiếng Việt (cần Internet)\n"
            f"Mic: {self.microphone_choice.get()}\n{self.voice_feedback}")
        self.voice_button.configure(
            text="Stop voice (F8)" if self.voice.active.is_set() else "Start voice (F8)")
        self._update_status()

    def start_eye_calibration(self):
        if not self.running:
            return
        self.eye_feature_history.clear()
        self.calibration = EyeCalibration()
        self.calibration_values = {
            "open": [[], []], "closed": [[], []],
            "blink_open": [[], []], "blink_closed": [[], []],
        }
        self.calibration_stage = "open"
        self.calibration_started = time.monotonic()
        self.calibration_stage_started = self.calibration_started
        self.calibration_closed_samples = 0
        self.auto_baseline_started = time.monotonic()
        self.auto_baseline_samples = {"ear": [[], []], "blink": [[], []]}
        self.error_status = ""
        self.gestures_armed = False
        self.eye_tracking_status = "EYE SETUP — gestures are suspended"
        self.gestures.lost_face()
        self._release_buttons()
        self._update_status()

    def toggle_enhancement(self):
        if not self.running:
            return
        self.enhancer.enabled = not self.enhancer.enabled
        self._update_status()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.gestures.lost_face()
        self.gestures_armed = False
        self._release_buttons()
        self._update_status()

    def _register_hotkeys(self):
        if os.name != "nt":
            return
        try:
            user32 = ctypes.windll.user32
            # MOD_NOREPEAT prevents key auto-repeat from toggling repeatedly.
            hotkeys = ((1, 0x77), (2, 0x78))  # F8, F9
            registered = []
            for hotkey_id, key in hotkeys:
                if not user32.RegisterHotKey(None, hotkey_id, 0x4000, key):
                    raise OSError(f"RegisterHotKey failed for virtual key {key}")
                registered.append(hotkey_id)
            class MSG(ctypes.Structure):
                _fields_ = [
                    ("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                    ("wParam", ctypes.c_size_t), ("lParam", ctypes.c_ssize_t),
                    ("time", ctypes.c_uint), ("pt_x", ctypes.c_long),
                    ("pt_y", ctypes.c_long),
                ]
            self._msg = MSG()
            self._registered_hotkeys = True
            self.root.after(80, self._poll_hotkeys)
        except Exception as exc:
            for hotkey_id in locals().get("registered", []):
                try:
                    ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
                except Exception:
                    pass
            self.gyro_status += f"; global hotkeys unavailable ({exc})"

    def _poll_hotkeys(self):
        if not self.running or not self._registered_hotkeys:
            return
        user32 = ctypes.windll.user32
        while user32.PeekMessageW(ctypes.byref(self._msg), None, 0x0312, 0x0312, 1):
            if self._msg.wParam == 1:
                self.toggle_voice()
            elif self._msg.wParam == 2:
                self.toggle_pause()
        self.root.after(80, self._poll_hotkeys)

    def _classify_eye(self, ear, side, previous, blink_score=None):
        if not self.calibration.ready:
            return "UNKNOWN"
        blink_threshold = (
            self.calibration.blink_threshold_left if side == "left"
            else self.calibration.blink_threshold_right)
        open_baseline = (self.calibration.open_left if side == "left"
                         else self.calibration.open_right)
        ear_threshold = (self.calibration.threshold_left if side == "left"
                         else self.calibration.threshold_right)
        ear_state = None
        if ear is not None and math.isfinite(ear):
            hysteresis = min(EYE_HYSTERESIS, max(0.004, open_baseline * 0.04))
            lower = ear_threshold - hysteresis
            upper = ear_threshold + hysteresis
            if ear < lower:
                ear_state = "CLOSED"
            elif ear > upper:
                ear_state = "OPEN"
            else:
                ear_state = None

        if blink_score is not None and blink_threshold is not None:
            blink_state = None
            if math.isfinite(blink_score):
                if blink_score > blink_threshold + BLENDSHAPE_STATE_MARGIN:
                    blink_state = "CLOSED"
                elif blink_score < blink_threshold - BLENDSHAPE_STATE_MARGIN:
                    blink_state = "OPEN"
            if ear_state is None:
                return blink_state if blink_state is not None else previous
            if blink_state is None:
                return ear_state
            if blink_state == ear_state:
                return blink_state
            if previous in ("OPEN", "CLOSED"):
                return previous
            return "UNKNOWN"

        if ear_state is not None:
            return ear_state
        if blink_score is not None and blink_threshold is not None:
            if not math.isfinite(blink_score):
                return "UNKNOWN"
            if previous in ("OPEN", "CLOSED"):
                return previous
            return "CLOSED" if blink_score >= blink_threshold else "OPEN"
        return "UNKNOWN"

    def record_eye_feedback(self, eye, result):
        if eye not in ("left", "right") or result not in (
            "success", "missed", "false_positive"):
            raise ValueError("Invalid eye feedback")
        if not self.calibration.ready or self.calibration_stage != "done":
            self.eye_feedback_status = "Hoàn tất thiết lập mắt trước khi ghi phản hồi"
            return

        now = time.monotonic()
        eye_index = 0 if eye == "left" else 1
        samples = [
            sample for sample in self.eye_feature_history
            if now - sample["time"] <= EYE_FEEDBACK_WINDOW_SECONDS
            and sample["ears"][eye_index] is not None
            and math.isfinite(sample["ears"][eye_index])
        ]
        if len(samples) < 3:
            self.eye_feedback_status = (
                "Chưa đủ số đo gần đây; giữ mặt trong khung hình rồi thử lại")
            return

        closed_samples = [sample for sample in samples
                          if sample["states"][eye] == "CLOSED"]
        candidate_samples = (
            closed_samples if result == "false_positive" and closed_samples
            else samples)
        ear_values = [sample["ears"][eye_index] for sample in samples]
        candidate_ear = min(
            sample["ears"][eye_index] for sample in candidate_samples)
        blink_values = [
            sample["blink_scores"][eye_index] for sample in samples
            if sample["blink_scores"][eye_index] is not None
            and math.isfinite(sample["blink_scores"][eye_index])
        ]
        candidate_blink = None
        open_blink_values = [
            sample["blink_scores"][eye_index] for sample in samples
            if sample["states"][eye] == "OPEN"
            and sample["blink_scores"][eye_index] is not None
            and math.isfinite(sample["blink_scores"][eye_index])
        ]

        open_baseline = (
            self.calibration.open_left if eye == "left"
            else self.calibration.open_right)
        ear_attribute = "threshold_left" if eye == "left" else "threshold_right"
        blink_attribute = (
            "blink_threshold_left" if eye == "left"
            else "blink_threshold_right")
        old_ear_threshold = getattr(self.calibration, ear_attribute)
        old_blink_threshold = getattr(self.calibration, blink_attribute)
        new_ear_threshold = old_ear_threshold
        new_blink_threshold = old_blink_threshold

        if result == "missed":
            ear_target = (open_baseline + candidate_ear) / 2
            if ear_target > old_ear_threshold:
                new_ear_threshold = min(
                    open_baseline * 0.92,
                    old_ear_threshold + EYE_FEEDBACK_EAR_STEP,
                    old_ear_threshold
                    + (ear_target - old_ear_threshold) * 0.2)
            if blink_values and open_blink_values:
                open_blink_baseline = float(np.median(open_blink_values))
                candidate_blink = max(
                    sample["blink_scores"][eye_index]
                    for sample in samples
                    if sample["blink_scores"][eye_index] is not None
                    and math.isfinite(sample["blink_scores"][eye_index]))
                if candidate_blink > open_blink_baseline + 0.08:
                    initial = old_blink_threshold
                    if initial is None:
                        initial = min(
                            0.95,
                            open_blink_baseline + AUTO_BASELINE_BLINK_MARGIN)
                    blink_target = (
                        open_blink_baseline + candidate_blink) / 2
                    if blink_target < initial:
                        new_blink_threshold = max(
                            open_blink_baseline + BLENDSHAPE_STATE_MARGIN,
                            initial - EYE_FEEDBACK_BLENDSHAPE_STEP,
                            initial + (blink_target - initial) * 0.2)
        elif result == "false_positive":
            ear_hysteresis = min(
                EYE_HYSTERESIS, max(0.004, open_baseline * 0.04))
            ear_target = max(
                open_baseline * 0.60,
                candidate_ear - ear_hysteresis * 1.5)
            if ear_target < old_ear_threshold:
                new_ear_threshold = max(
                    open_baseline * 0.60,
                    old_ear_threshold - EYE_FEEDBACK_EAR_STEP,
                    old_ear_threshold + (ear_target - old_ear_threshold) * 0.2)
            if blink_values and open_blink_values:
                open_blink_baseline = float(np.median(open_blink_values))
                closed_blink_values = [
                    sample["blink_scores"][eye_index]
                    for sample in candidate_samples
                    if sample["blink_scores"][eye_index] is not None
                    and math.isfinite(sample["blink_scores"][eye_index])
                ]
                if closed_blink_values:
                    candidate_blink = max(closed_blink_values)
                    initial = old_blink_threshold
                    if initial is None:
                        initial = min(
                            0.95,
                            open_blink_baseline + AUTO_BASELINE_BLINK_MARGIN)
                    blink_target = min(
                        0.95,
                        candidate_blink + BLENDSHAPE_STATE_MARGIN * 1.5)
                    if blink_target > initial:
                        new_blink_threshold = min(
                            0.95,
                            initial + EYE_FEEDBACK_BLENDSHAPE_STEP,
                            initial + (blink_target - initial) * 0.2)

        feedback_record = {
            "timestamp": time.time(),
            "eye": eye,
            "result": result,
            "sample_count": len(samples),
            "ear_min": min(ear_values),
            "ear_median": float(np.median(ear_values)),
            "ear_max": max(ear_values),
            "blink_min": min(blink_values) if blink_values else None,
            "blink_max": max(blink_values) if blink_values else None,
            "eye_width_px": self.last_eye_widths[eye_index],
            "camera_resolution": self.camera_resolution,
            "threshold_before": {
                "ear": old_ear_threshold,
                "blink": old_blink_threshold,
            },
            "threshold_after": {
                "ear": new_ear_threshold,
                "blink": new_blink_threshold,
            },
        }
        try:
            os.makedirs(os.path.dirname(EYE_FEEDBACK_FILE), exist_ok=True)
            with open(EYE_FEEDBACK_FILE, "a", encoding="utf-8") as feedback_file:
                feedback_file.write(json.dumps(feedback_record) + "\n")
        except OSError as exc:
            self.error_status = f"Could not save eye feedback: {exc}"
            self.eye_feedback_status = "Phản hồi chưa được lưu; ngưỡng không thay đổi"
            return

        setattr(self.calibration, ear_attribute, new_ear_threshold)
        setattr(self.calibration, blink_attribute, new_blink_threshold)
        if (new_ear_threshold != old_ear_threshold
                or new_blink_threshold != old_blink_threshold):
            self._save_eye_calibration()
            adjustment = "đã tinh chỉnh ngưỡng một bước nhỏ"
        else:
            adjustment = "ghi nhận; ngưỡng giữ nguyên"
        self.eye_feedback_count += 1
        result_text = {
            "success": "đúng",
            "missed": "bỏ sót",
            "false_positive": "nhận nhầm",
        }[result]
        self.eye_feedback_status = (
            f"{eye} · {result_text} · {adjustment} "
            f"(lần {self.eye_feedback_count})")

    def _update_auto_eye_baseline(self, ears, blink_scores, now):
        if now - self.auto_baseline_started > AUTO_BASELINE_TIMEOUT_SECONDS:
            if self.calibration.ready:
                self.calibration_stage = "done"
                self.error_status = (
                    "Auto baseline did not settle; keeping saved eye settings.")
            else:
                self.calibration_stage = "open"
                self.calibration_started = now
                self.calibration_stage_started = now
                self.calibration_values = {
                    "open": [[], []], "closed": [[], []],
                    "blink_open": [[], []], "blink_closed": [[], []],
                }
                self.error_status = (
                    "Auto eye setup needs a steady open-eye view. "
                    "Use R to run the guided fallback.")
            return

        valid_ears = all(
            value is not None and math.isfinite(value) and value > 0.04
            for value in ears)
        valid_blinks = (
            blink_scores is not None
            and all(score is not None and math.isfinite(score)
                    and score <= AUTO_BASELINE_CLOSED_SCORE_MAX
                    for score in blink_scores))
        if not valid_ears or (blink_scores is not None and not valid_blinks):
            self.auto_baseline_samples = {"ear": [[], []], "blink": [[], []]}
            return

        for index, value in enumerate(ears):
            self.auto_baseline_samples["ear"][index].append(value)
            self.auto_baseline_samples["ear"][index] = (
                self.auto_baseline_samples["ear"][index][-AUTO_BASELINE_MIN_SAMPLES:])
            if valid_blinks:
                score = blink_scores[index]
                self.auto_baseline_samples["blink"][index].append(score)
                self.auto_baseline_samples["blink"][index] = (
                    self.auto_baseline_samples["blink"][index]
                    [-AUTO_BASELINE_MIN_SAMPLES:])

        if any(len(values) < AUTO_BASELINE_MIN_SAMPLES
               for values in self.auto_baseline_samples["ear"]):
            return

        opened, thresholds, blink_thresholds = [], [], []
        for index in (0, 1):
            samples = self.auto_baseline_samples["ear"][index]
            baseline = float(np.median(samples))
            if baseline <= 0 or float(np.std(samples)) / baseline > 0.20:
                return
            opened.append(baseline)
            if self.calibration.ready:
                old_open = (
                    self.calibration.open_left if index == 0
                    else self.calibration.open_right)
                old_threshold = (
                    self.calibration.threshold_left if index == 0
                    else self.calibration.threshold_right)
                learned_ratio = clamp(old_threshold / old_open, 0.60, 0.95)
                thresholds.append(baseline * learned_ratio)
            else:
                thresholds.append(
                    baseline * AUTO_BASELINE_EAR_THRESHOLD_RATIO)

            blink_samples = self.auto_baseline_samples["blink"][index]
            if len(blink_samples) >= AUTO_BASELINE_MIN_SAMPLES:
                blink_baseline = float(np.median(blink_samples))
                fresh_threshold = min(
                    0.95, blink_baseline + AUTO_BASELINE_BLINK_MARGIN)
                if self.calibration.ready:
                    old_blink_threshold = (
                        self.calibration.blink_threshold_left if index == 0
                        else self.calibration.blink_threshold_right)
                    blink_thresholds.append(
                        old_blink_threshold
                        if old_blink_threshold is not None else fresh_threshold)
                else:
                    blink_thresholds.append(fresh_threshold)
            else:
                old_blink_threshold = (
                    self.calibration.blink_threshold_left if index == 0
                    else self.calibration.blink_threshold_right)
                blink_thresholds.append(
                    old_blink_threshold if self.calibration.ready else None)

        self.calibration = EyeCalibration(
            open_left=opened[0], open_right=opened[1],
            threshold_left=thresholds[0], threshold_right=thresholds[1],
            blink_threshold_left=blink_thresholds[0],
            blink_threshold_right=blink_thresholds[1],
            ready=True)
        self.calibration_stage = "done"
        self.error_status = ""
        self._save_eye_calibration()

    def _adapt_open_eye_baseline(self, ears, blink_scores):
        if (not self.calibration.ready
                or self.calibration_stage != "done"
                or self.eye_state["left"] != "OPEN"
                or self.eye_state["right"] != "OPEN"
                or self.gestures.dragging or self.scroll_mode):
            return

        opened = [self.calibration.open_left, self.calibration.open_right]
        thresholds = [self.calibration.threshold_left,
                      self.calibration.threshold_right]
        blink_thresholds = [self.calibration.blink_threshold_left,
                            self.calibration.blink_threshold_right]
        changed = False
        for index, value in enumerate(ears):
            if value is None or not math.isfinite(value):
                continue
            if not opened[index] * 0.88 <= value <= opened[index] * 1.12:
                continue
            threshold_ratio = thresholds[index] / opened[index]
            if blink_scores is not None:
                score = blink_scores[index]
                if score is None or not math.isfinite(score):
                    continue
                if (blink_thresholds[index] is not None
                        and score > blink_thresholds[index] - 0.08):
                    continue
                if blink_thresholds[index] is None:
                    blink_thresholds[index] = min(
                        0.95, score + AUTO_BASELINE_BLINK_MARGIN)
            opened[index] += (value - opened[index]) * AUTO_BASELINE_ADAPT_RATE
            thresholds[index] = opened[index] * threshold_ratio
            changed = True

        if changed:
            self.calibration = EyeCalibration(
                open_left=opened[0], open_right=opened[1],
                threshold_left=thresholds[0], threshold_right=thresholds[1],
                blink_threshold_left=blink_thresholds[0],
                blink_threshold_right=blink_thresholds[1], ready=True)

    def _eye_samples(self, ears, now, blink_scores=None):
        target = None
        if self.calibration_stage == "open":
            if now - self.calibration_started < CALIBRATION_OPEN_SECONDS:
                target = "open"
            else:
                self.calibration_stage = "left"
                self.calibration_stage_started = now
                target = None
        elif self.calibration_stage in ("left", "right"):
            if now - self.calibration_stage_started > CALIBRATION_BLINK_TIMEOUT_SECONDS:
                eye = self.calibration_stage.upper()
                self.calibration_stage = "error"
                self.error_status = (
                    f"Calibration timed out: gently close only your {eye} eye, "
                    "then reopen it; "
                    "press R to retry.")
                return
            target_index = 0 if self.calibration_stage == "left" else 1
            other_index = 1 - target_index
            open_values = self.calibration_values["open"]
            if not open_values[0] or not open_values[1]:
                return
            if blink_scores is not None and all(
                    self.calibration_values["blink_open"][index]
                    for index in (0, 1)):
                blink_open = self.calibration_values["blink_open"]
                baselines = [float(np.median(values)) for values in blink_open]
                target_closed = (
                    blink_scores[target_index] is not None
                    and blink_scores[target_index]
                    >= max(0.12, baselines[target_index]
                           + CALIBRATION_BLINK_CLOSED_DELTA))
                other_open = (
                    blink_scores[other_index] is not None
                    and blink_scores[other_index]
                    <= max(0.45, baselines[other_index]
                           + CALIBRATION_BLINK_OTHER_EYE_DELTA))
            else:
                open_baselines = [float(np.median(values)) for values in open_values]
                target_closed = (
                    ears[target_index] is not None
                    and ears[target_index]
                    <= open_baselines[target_index] * CALIBRATION_CLOSED_RATIO)
                other_open = (
                    ears[other_index] is not None
                    and ears[other_index]
                    >= open_baselines[other_index] * CALIBRATION_OTHER_EYE_OPEN_RATIO)
            if target_closed and other_open:
                closed_value = ears[target_index]
                if closed_value is not None:
                    self.calibration_values["closed"][target_index].append(closed_value)
                if blink_scores is not None and blink_scores[target_index] is not None:
                    self.calibration_values["blink_closed"][target_index].append(
                        blink_scores[target_index])
                self.calibration_closed_samples += 1
            elif (self.calibration_closed_samples >= CALIBRATION_MIN_CLOSED_SAMPLES
                  and not target_closed):
                if self.calibration_stage == "left":
                    self.calibration_stage = "right"
                    self.calibration_stage_started = now
                    self.calibration_closed_samples = 0
                else:
                    self._finish_eye_calibration()
                return
            else:
                self.calibration_closed_samples = 0
        else:
            return

        if target == "open":
            for index, value in enumerate(ears):
                if value is not None:
                    self.calibration_values["open"][index].append(value)
                if blink_scores is not None and blink_scores[index] is not None:
                    self.calibration_values["blink_open"][index].append(
                        blink_scores[index])

    def _finish_eye_calibration(self):
        open_values = self.calibration_values["open"]
        closed_values = self.calibration_values["closed"]
        opens, thresholds, blink_thresholds = [], [], []
        for index in (0, 1):
            open_sample = open_values[index]
            closed_sample = closed_values[index]
            blink_open = self.calibration_values["blink_open"][index]
            blink_closed = self.calibration_values["blink_closed"][index]
            has_blink_calibration = (
                len(blink_open) >= 8
                and len(blink_closed) >= CALIBRATION_MIN_CLOSED_SAMPLES)
            if (len(open_sample) < 8
                    or (len(closed_sample) < CALIBRATION_MIN_CLOSED_SAMPLES
                        and not has_blink_calibration)):
                self.calibration_stage = "error"
                eye = "left" if index == 0 else "right"
                self.error_status = (
                    f"Calibration failed for {eye} eye. Keep your face centered, "
                    "gently close that eye alone, then press R to retry.")
                return
            opened = float(np.percentile(open_sample, 20))
            closed = float(np.percentile(closed_sample, 80)) if closed_sample else 0.0
            blink_threshold = None
            if has_blink_calibration:
                blink_opened = float(np.percentile(blink_open, 80))
                blink_closed_value = float(np.percentile(blink_closed, 20))
                if blink_closed_value > blink_opened + 0.08:
                    blink_threshold = (blink_opened + blink_closed_value) / 2
            if blink_threshold is None and (
                    not closed_sample
                    or opened <= closed + max(0.015, opened * 0.08)):
                self.calibration_stage = "error"
                self.error_status = (
                    "Eye calibration could not separate open and closed states. "
                    "Keep your face centered and try R again.")
                return
            threshold = (clamp((opened + closed) / 2, 0.10, opened * 0.9)
                         if closed_sample and opened > closed else opened * 0.65)
            opens.append(opened)
            thresholds.append(threshold)
            blink_thresholds.append(blink_threshold)
        self.calibration = EyeCalibration(
            open_left=opens[0], open_right=opens[1],
            threshold_left=thresholds[0], threshold_right=thresholds[1],
            blink_threshold_left=blink_thresholds[0],
            blink_threshold_right=blink_thresholds[1], ready=True)
        self.calibration_stage = "done"
        if self.error_status.startswith("Saved eye thresholds"):
            self.error_status = ""
        self._save_eye_calibration()

    def _save_eye_calibration(self):
        values = {
            "version": 3,
            "open_left": self.calibration.open_left,
            "open_right": self.calibration.open_right,
            "threshold_left": self.calibration.threshold_left,
            "threshold_right": self.calibration.threshold_right,
            "blink_threshold_left": self.calibration.blink_threshold_left,
            "blink_threshold_right": self.calibration.blink_threshold_right,
        }
        try:
            os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
            temporary_file = CALIBRATION_FILE + ".tmp"
            with open(temporary_file, "w", encoding="utf-8") as calibration_file:
                json.dump(values, calibration_file, indent=2)
            os.replace(temporary_file, CALIBRATION_FILE)
        except OSError as exc:
            self.error_status = (
                f"Calibration works, but thresholds could not be saved: {exc}")

    def _face_measurements(self, result, width, height):
        try:
            landmarks = result.face_landmarks[0]
        except (AttributeError, IndexError, TypeError):
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            coordinates = [(float(point.x), float(point.y))
                           for point in landmarks]
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        required_landmark_count = max(
            max(pair) for contour in EYE_CONTOURS
            for pair in (contour[0], *contour[1]))
        required_landmark_count = max(
            required_landmark_count,
            max(index for points_for_eye in EYE_DRAW_POINTS
                for index in points_for_eye) + 1,
            max(FACE_OVAL) + 1)
        if len(coordinates) < required_landmark_count:
            return None
        normalized = np.asarray(coordinates, dtype=np.float32)
        if not np.isfinite(normalized).all():
            return None
        points = normalized * np.asarray((width, height), dtype=np.float32)
        measured = []
        for contour in EYE_CONTOURS:
            measured.append((eye_aspect_ratio(points, contour),
                             float(points[list(contour[0])].mean(axis=0)[0]),
                             float(np.linalg.norm(
                                 points[contour[0][0]] - points[contour[0][1]]))))
        # User's left eye is on viewer-left in the mirrored preview.
        measured.sort(key=lambda entry: entry[1])
        ears = (measured[0][0], measured[1][0])
        centers = (measured[0][1], measured[1][1])
        eye_widths = (measured[0][2], measured[1][2])
        draw = []
        eye_trackable = []
        for contour_index, indices in enumerate(EYE_DRAW_POINTS):
            center = float(points[list(EYE_CONTOURS[contour_index][0])].mean(axis=0)[0])
            draw.append((center, points[list(indices)]))
            eye_points = normalized[list(indices)]
            eye_width = float(np.linalg.norm(
                points[EYE_CONTOURS[contour_index][0][0]]
                - points[EYE_CONTOURS[contour_index][0][1]]))
            ear = eye_aspect_ratio(points, EYE_CONTOURS[contour_index])
            in_frame = bool(
                np.all((eye_points[:, 0] >= 0.0) & (eye_points[:, 0] <= 1.0))
                and np.all((eye_points[:, 1] >= 0.0) & (eye_points[:, 1] <= 1.0)))
            landmark_scores = []
            invalid_confidence = False
            for landmark_index in indices:
                landmark = landmarks[landmark_index]
                for attribute in ("visibility", "presence"):
                    score = getattr(landmark, attribute, None)
                    if score is not None:
                        try:
                            score = float(score)
                        except (TypeError, ValueError, OverflowError):
                            invalid_confidence = True
                            continue
                        if math.isfinite(score):
                            landmark_scores.append(score)
                        else:
                            invalid_confidence = True
            confidence_ok = (
                not invalid_confidence
                and (not landmark_scores
                     or min(landmark_scores) >= MIN_EYE_LANDMARK_VISIBILITY))
            eye_trackable.append((
                center,
                eye_width >= MIN_TRACKABLE_EYE_WIDTH_PIXELS
                and ear is not None
                and math.isfinite(ear)
                and in_frame
                and confidence_ok))
        blink_scores = None
        try:
            categories = result.face_blendshapes[0]
            scores_by_name = {category.category_name: float(category.score)
                              for category in categories}
            # Blendshape sides are inverted for a mirrored input image:
            # viewer-left therefore uses the subject's eyeBlinkRight score.
            blink_scores = (
                scores_by_name.get("eyeBlinkRight"),
                scores_by_name.get("eyeBlinkLeft"),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        eye_trackable_by_side = tuple(
            trackable for _center, trackable in sorted(
                eye_trackable, key=lambda entry: entry[0]))
        face_outline = points[list(FACE_OVAL)].astype(np.int32)
        return (
            ears, centers, draw, blink_scores, eye_widths,
            eye_trackable_by_side, face_outline)

    def _draw_eye_landmarks(self, frame, draw):
        scale = max(1.0, frame.shape[1] / 640)
        radius = max(2, round(2 * scale))
        thickness = max(1, round(2 * scale))
        for side_index, (_center, coords) in enumerate(sorted(draw, key=lambda entry: entry[0])):
            side = "LEFT" if side_index == 0 else "RIGHT"
            for point in coords:
                cv2.circle(
                    frame, (int(point[0]), int(point[1])), radius,
                    (30, 240, 100), -1)
            anchor = tuple(np.mean(coords, axis=0).astype(int))
            cv2.putText(frame, side, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * scale, (30, 240, 100), thickness, cv2.LINE_AA)

    def _draw_face_outline(self, frame, outline):
        scale = max(1.0, frame.shape[1] / 640)
        cv2.polylines(
            frame, [outline.reshape((-1, 1, 2))], True, (0, 0, 255),
            max(2, round(2 * scale)), cv2.LINE_AA)

    def _update_eye_simulation(self):
        canvas = self.eye_simulation
        canvas.delete("all")
        width = max(300, canvas.winfo_width())
        height = max(140, canvas.winfo_height())
        center_x = width / 2
        center_y = height / 2 + 5
        face_color = "#ed3b3b" if self.face_detected else "#687582"
        canvas.create_oval(
            center_x - 105, center_y - 72, center_x + 105, center_y + 72,
            fill="#f3cfac", outline=face_color, width=4)
        canvas.create_text(
            center_x, 13,
            text="ĐỎ = ĐÃ PHÁT HIỆN MẶT" if self.face_detected else "CHƯA THẤY MẶT",
            fill=face_color, font=("Segoe UI", 9, "bold"))
        eye_centers = (center_x - 48, center_x + 48)
        widths = self.last_eye_widths
        valid_widths = [value for value in widths if value is not None and value > 0]
        mean_width = float(np.mean(valid_widths)) if valid_widths else 1.0
        for index, side in enumerate(("left", "right")):
            baseline = (
                self.calibration.open_left if index == 0
                else self.calibration.open_right)
            ear = self.last_ears[index]
            state = self.eye_state[side]
            eye_width = widths[index]
            relative_width = (
                clamp(eye_width / mean_width, 0.75, 1.25)
                if eye_width is not None and mean_width > 0 else 1.0)
            half_width = 28 * relative_width
            ratio = (
                clamp(ear / baseline, 0.0, 1.0)
                if ear is not None and baseline > 0 else 0.0)
            eye_height = max(2.0, 16.0 * ratio)
            color = (
                "#75e0a1" if state == "OPEN"
                else "#ffcf66" if state == "CLOSED"
                else "#82909d")
            eye_x = eye_centers[index]
            canvas.create_oval(
                eye_x - half_width, center_y - eye_height,
                eye_x + half_width, center_y + eye_height,
                fill="#ffffff" if ratio > 0.08 else "#303840",
                outline=color, width=3)
            if ratio > 0.08:
                canvas.create_oval(
                    eye_x - 4, center_y - 4, eye_x + 4, center_y + 4,
                    fill="#20252b", outline="")
            label = "L" if index == 0 else "R"
            canvas.create_text(
                eye_x, center_y + 37,
                text=f"{label} {state} · {ratio:.0%}",
                fill=color, font=("Segoe UI", 9, "bold"))
        if self.calibration.ready:
            canvas.create_text(
                center_x, height - 10,
                text=(f"Ngưỡng riêng: L {self.calibration.threshold_left:.3f}  "
                      f"R {self.calibration.threshold_right:.3f}"),
                fill="#d0d8e0", font=("Consolas", 9))
        else:
            canvas.create_text(
                center_x, height - 10, text="Chưa có baseline cá nhân",
                fill="#d0d8e0", font=("Segoe UI", 9))

    def _mouse_action(self, action, button):
        try:
            if action == "click":
                pyautogui.click(button=button)
                self.last_command = f"{button} click"
            elif action == "multi_click":
                count = max(1, min(2, int(button)))
                pyautogui.click(button="left", clicks=count, interval=0.08)
                self.last_command = f"left ×{count} click"
            elif action == "down":
                pyautogui.mouseDown(button=button)
                self.last_mouse_button = button
                self.last_command = f"{button} down — drag"
            elif action == "up":
                pyautogui.mouseUp(button=button)
                self.last_mouse_button = None
                self.last_command = f"{button} up"
        except pyautogui.FailSafeException:
            self.error_status = "Mouse failsafe triggered; gestures paused."
            self.toggle_pause()
        except Exception as exc:
            self.error_status = f"Mouse action failed: {exc}"
            self._release_buttons()

    def _release_buttons(self, force=False):
        buttons = ("left", "right", "middle") if force else (
            (self.last_mouse_button,) if self.last_mouse_button else ())
        for button in buttons:
            try:
                pyautogui.mouseUp(button=button)
            except Exception:
                pass
        self.last_mouse_button = None

    def _process_gyro(self, now):
        sample = self.gyro.snapshot()
        dt = clamp(now - self.last_gyro_tick, 0.0, 0.1)
        self.last_gyro_tick = now
        if sample is None:
            return
        x, y, timestamp, neutral = sample
        if neutral is None:
            return
        if now - timestamp > GYRO_STALE_SECONDS:
            self.gyro_status = "WARNING — gyro sample stale"
            return
        if self.gyro_status == "WARNING — gyro sample stale":
            self.gyro_status = "connected; neutral calibrated"
        x -= neutral[0]
        y -= neutral[1]
        if abs(x) < GYRO_DEADZONE_DPS:
            x = 0.0
        if abs(y) < GYRO_DEADZONE_DPS:
            y = 0.0
        if self.scroll_mode:
            axis = y if GYRO_SCROLL_AXIS == "y" else x
            self.scroll_fraction += axis * dt * GYRO_SCROLL_SENSITIVITY * SCROLL_AXIS_SIGN
            steps = int(self.scroll_fraction)
            if steps:
                try:
                    pyautogui.scroll(steps)
                    self.scroll_fraction -= steps
                except Exception as exc:
                    self.error_status = f"Scroll failed: {exc}"
            return
        if self.paused:
            return
        screen_w, screen_h = pyautogui.size()
        target_x = clamp(
            self.filtered_pointer[0] + x * dt * GYRO_POINTER_SENSITIVITY, 0, screen_w - 1)
        target_y = clamp(
            self.filtered_pointer[1] + y * dt * GYRO_POINTER_SENSITIVITY, 0, screen_h - 1)
        self.filtered_pointer[0] = (
            (1 - POINTER_SMOOTHING) * self.filtered_pointer[0]
            + POINTER_SMOOTHING * target_x)
        self.filtered_pointer[1] = (
            (1 - POINTER_SMOOTHING) * self.filtered_pointer[1]
            + POINTER_SMOOTHING * target_y)
        try:
            pyautogui.moveTo(int(self.filtered_pointer[0]), int(self.filtered_pointer[1]),
                             duration=0)
        except pyautogui.FailSafeException:
            self.error_status = "Mouse failsafe triggered; gestures paused."
            self.toggle_pause()
        except Exception as exc:
            self.error_status = f"Pointer movement failed: {exc}"

    def _insert_text(self, text, generation):
        if (not self.running or not text
                or not self.voice.is_active_session(generation)):
            return
        if self.voice_to_panel.get():
            self.voice_text.insert("end", text + " ")
            self.voice_text.see("end")
            self.last_command = f"dictated to test panel: {text[:48]}"
            self.voice_feedback = f"Đã nhận: {text}"
            self.voice_panel_status.set(
                f"Voice: {self.voice_status} · nhận dạng tiếng Việt (cần Internet)\n"
                f"Mic: {self.microphone_choice.get()}\n{self.voice_feedback}")
            return
        # Paste Unicode via the Windows clipboard; restore prior text afterwards.
        previous = None
        try:
            try:
                previous = self.root.clipboard_get()
            except tk.TclError:
                pass
            self.root.clipboard_clear()
            self.root.clipboard_append(text + " ")
            self.root.update_idletasks()
            pyautogui.hotkey("ctrl", "v")
            self.last_command = f"dictated: {text[:48]}"
        except Exception as exc:
            self.error_status = f"Text insertion failed: {exc}"
            self.voice_feedback = f"Nhận dạng được nhưng không ghi vào ô thử: {exc}"
            self.voice_panel_status.set(
                f"Voice: {self.voice_status} · nhận dạng tiếng Việt\n"
                f"Mic: {self.microphone_choice.get()}\n{self.voice_feedback}")
            return
        finally:
            if previous is not None:
                self.root.after(700, lambda value=previous: self._restore_clipboard(value))

    def _restore_clipboard(self, value):
        if not self.running:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
        except tk.TclError:
            pass

    def _update_status(self):
        if self.paused:
            mode = "PAUSED"
        elif self.gestures.dragging:
            mode = "DRAG"
        elif self.scroll_mode:
            mode = "SCROLL_MODE"
        elif self.voice.active.is_set():
            mode = "VOICE_DICTATION"
        else:
            mode = "NORMAL"
        self.last_mode = mode
        color = "#ffcf66" if self.paused or self.scroll_mode else "#75e0a1"
        self.mode_label.configure(text=mode, fg=color)
        if self.paused:
            eye_tracking = "PAUSED — press F9 to resume"
        elif self.calibration_stage != "done":
            eye_tracking = "EYE SETUP — gestures are suspended"
        elif not self.face_detected:
            eye_tracking = "NO FACE — gestures stopped; show your face to resume"
        elif self.eye_tracking_status != "TRACKING":
            eye_tracking = self.eye_tracking_status
        elif not self.gestures_armed:
            eye_tracking = "FACE SEEN — waiting for both eyes OPEN to resume"
        else:
            eye_tracking = "ACTIVE — eye gestures are enabled"
        if self.calibration_stage == "auto":
            calibration_guide = (
                "Đặt mặt giữa khung hình, ở khoảng cách và ánh sáng thường dùng. "
                "Nhìn tự nhiên, giữ hai mắt mở khoảng 2 giây để lấy baseline riêng. "
                "Đây là mô phỏng số đo, không lưu ảnh/video. Muốn hiệu chuẩn có "
                "hướng dẫn cho từng mắt, bấm nút bên dưới.")
        elif self.calibration_stage == "open":
            calibration_guide = (
                "Bước 1/3 · Ngồi ở khoảng cách/ánh sáng thường dùng, nhìn thẳng "
                "và giữ hai mắt mở tự nhiên trong 2 giây. Không cần mở mắt quá to.")
        elif self.calibration_stage == "left":
            calibration_guide = (
                "Bước 2/3 · Khi đã sẵn sàng, nhẹ nhàng nhắm rồi mở mắt ở BÊN TRÁI "
                "khung hình xem trước. Không cần nheo mạnh; lặp lại đến khi bước "
                "tự chuyển. Mắt còn lại giữ mở.")
        elif self.calibration_stage == "right":
            calibration_guide = (
                "Bước 3/3 · Nhẹ nhàng nhắm rồi mở mắt ở BÊN PHẢI khung hình xem "
                "trước. Mắt còn lại giữ mở. Khi hoàn tất, kiểm tra mô phỏng.")
        elif self.calibration_stage == "error":
            calibration_guide = (
                f"Thiết lập chưa hoàn tất: {self.error_status} "
                "Điều chỉnh vị trí/ánh sáng rồi bấm nút hiệu chuẩn lại.")
        else:
            calibration_guide = (
                "Hiệu chuẩn xong. Hình mô phỏng dùng EAR/open-baseline riêng từng "
                "mắt, nên tự chuẩn hóa độ mở cho mắt to/nhỏ. Thử nháy nhẹ từng mắt; "
                "nếu sai, dùng các nút phản hồi bên trái.")
        if self.eye_tracking_status.startswith("EYES NOT TRACKABLE"):
            calibration_guide += (
                "\nKhông thao tác mắt lúc này: đưa cả hai mắt trọn vào khung hình "
                "và ngồi gần hơn/đủ sáng cho landmark rõ.")
        elif not self.face_detected:
            calibration_guide += (
                "\nCamera chưa thấy khuôn mặt: hướng mặt về camera, bỏ vật che "
                "mắt và tăng ánh sáng trước mặt.")
        self.calibration_guide_text.set(calibration_guide)
        self._update_eye_simulation()
        auto_samples = min(
            len(self.auto_baseline_samples["ear"][0]),
            len(self.auto_baseline_samples["ear"][1]))
        calibration = ("FAILED — press R to retry" if self.calibration_stage == "error"
                       else f"AUTO SETUP: keep both eyes naturally open ({auto_samples}/{AUTO_BASELINE_MIN_SAMPLES})"
                       if self.calibration_stage == "auto"
                       else "Step 1/3: keep both eyes open" if self.calibration_stage == "open"
                       else "Step 2/3: blink LEFT eye only" if self.calibration_stage == "left"
                       else "Step 3/3: blink RIGHT eye only" if self.calibration_stage == "right"
                       else "READY — adaptive thresholds active; press R for guided setup")
        small_eyes = any(
            width is not None and width < SMALL_EYE_WIDTH_PIXELS
            for width in self.last_eye_widths)
        message = (
            f"MODE: {mode}\n"
            f"Eye tracking: {eye_tracking}\n"
            f"LEFT EYE:  {self.eye_state['left']}"
            f"  blink={self._format_ear(self.last_blink_scores[0])}"
            f"  EAR={self._format_ear(self.last_ears[0])}\n"
            f"RIGHT EYE: {self.eye_state['right']}"
            f"  blink={self._format_ear(self.last_blink_scores[1])}"
            f"  EAR={self._format_ear(self.last_ears[1])}\n\n"
            f"Eye gestures: close/reopen <= {QUICK_BLINK_MAX_SECONDS:.1f}s = click; "
            f"hold >= {DRAG_HOLD_SECONDS:.2f}s = drag/scroll; "
            f"other eye may blink during hold; both closed "
            f"{DOUBLE_CLOSED_EXIT_SECONDS:.0f}s exits hold; "
            f"{QUICK_BLINK_MAX_SECONDS:.1f}-{DRAG_HOLD_SECONDS:.2f}s = ignored; "
            f"double-click gap <= {BLINK_SEQUENCE_GAP_SECONDS:.2f}s\n"
            f"Camera: {self.camera_resolution[0]}x{self.camera_resolution[1]} · "
            f"eye width L/R: {self._format_ear(self.last_eye_widths[0])}/"
            f"{self._format_ear(self.last_eye_widths[1])} px"
            f"{' — eyes small, tracking may be less reliable' if small_eyes else ''}\n"
            f"Calibration: {calibration}\n"
            f"Feedback learning: {self.eye_feedback_status}\n"
            f"Low-light enhancement: {'ON' if self.enhancer.enabled else 'OFF'}\n"
            f"Pending command: {self._pending_label()}\n"
            f"Gyroscope: {self.gyro_status}\n"
            f"Microphone: {self.voice_status}\n"
            f"Frame rate: {self.fps:.1f} FPS\n"
            f"Last action: {self.last_command}\n"
            f"{self.error_status}"
        )
        self.status_text.set(message)

    @staticmethod
    def _format_ear(value):
        return "—" if value is None else f"{value:.3f}"

    def _pending_label(self):
        if self.gestures.scroll_mode:
            return "gyro scroll; hold RIGHT eye, open it to exit"
        pending = []
        for eye, data in self.gestures.pending.items():
            if data:
                pending.append(f"{eye} blink ×{data['count']}")
        if self.gestures.dragging:
            return "left button held — drag"
        if pending:
            return "waiting: " + ", ".join(pending)
        return "none"

    def _tick(self):
        if not self.running:
            return
        tick_started = time.monotonic()
        while True:
            try:
                self.callback_queue.get_nowait()()
            except queue.Empty:
                break
        now = time.monotonic()
        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.error_status = "Camera frame unavailable."
                self.face_detected = False
                self.eye_tracking_status = "CAMERA UNAVAILABLE — gestures stopped"
                self.eye_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
                self.last_ears = (None, None)
                self.last_eye_widths = (None, None)
                self.last_blink_scores = (None, None)
                self.eye_feature_history.clear()
                if self.calibration_stage == "auto":
                    self.auto_baseline_samples = {
                        "ear": [[], []], "blink": [[], []]}
                self.gestures.lost_face()
                self.gestures_armed = False
                self._release_buttons()
            else:
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                self.camera_resolution = (width, height)
                inference_frame = frame
                if width > MAX_INFERENCE_WIDTH:
                    inference_height = max(1, round(height * MAX_INFERENCE_WIDTH / width))
                    inference_frame = cv2.resize(
                        frame, (MAX_INFERENCE_WIDTH, inference_height),
                        interpolation=cv2.INTER_AREA)
                analysis_frame = self.enhancer.process(inference_frame)
                rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(image, int(now * 1000))
                measurements = self._face_measurements(result, width, height)
                if measurements is None:
                    self.face_detected = False
                    self.eye_tracking_status = "NO FACE — gestures stopped"
                    self.eye_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
                    self.last_ears = (None, None)
                    self.last_eye_widths = (None, None)
                    self.last_blink_scores = (None, None)
                    self.eye_feature_history.clear()
                    if self.calibration_stage == "auto":
                        self.auto_baseline_samples = {
                            "ear": [[], []], "blink": [[], []]}
                    self.gestures.lost_face()
                    self.gestures_armed = False
                    self._release_buttons()
                    cv2.putText(frame, "NO FACE - eye gestures suspended",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    self.face_detected = True
                    (ears, _centers, draw, blink_scores,
                     eye_widths, eye_trackable, face_outline) = measurements
                    self.last_ears = ears
                    self.last_eye_widths = eye_widths
                    self.last_blink_scores = blink_scores or (None, None)
                    self._draw_face_outline(frame, face_outline)
                    self._draw_eye_landmarks(frame, draw)
                    if not all(eye_trackable):
                        missing_sides = [
                            "L" if index == 0 else "R"
                            for index, trackable in enumerate(eye_trackable)
                            if not trackable
                        ]
                        self.eye_tracking_status = (
                            "EYES NOT TRACKABLE "
                            f"({','.join(missing_sides)}) — gestures stopped")
                        self.eye_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
                        self.eye_feature_history.clear()
                        if self.calibration_stage == "auto":
                            self.auto_baseline_samples = {
                                "ear": [[], []], "blink": [[], []]}
                        self.gestures.lost_face()
                        self.gestures_armed = False
                        self._release_buttons()
                        cv2.putText(
                            frame, "EYES NOT TRACKABLE - gestures suspended",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        self.eye_tracking_status = "TRACKING"
                        if self.calibration_stage == "auto":
                            self._update_auto_eye_baseline(
                                ears, blink_scores, now)
                        elif self.calibration_stage in ("open", "left", "right"):
                            self._eye_samples(ears, now, blink_scores)
                        left = self._classify_eye(
                            ears[0], "left", self.eye_state["left"],
                            self.last_blink_scores[0])
                        right = self._classify_eye(
                            ears[1], "right", self.eye_state["right"],
                            self.last_blink_scores[1])
                        self.eye_state = {"left": left, "right": right}
                        if (self.calibration.ready
                                and self.calibration_stage == "done"):
                            self.eye_feature_history.append({
                                "time": now,
                                "ears": ears,
                                "blink_scores": self.last_blink_scores,
                                "states": self.eye_state.copy(),
                            })
                        self._adapt_open_eye_baseline(ears, blink_scores)
                        if "UNKNOWN" in (left, right):
                            self.gestures.lost_face()
                            self.gestures_armed = False
                            self._release_buttons()
                        elif (self.calibration.ready
                              and self.calibration_stage == "done"
                              and not self.paused):
                            if not self.gestures_armed:
                                if left == "OPEN" and right == "OPEN":
                                    self.gestures.reset()
                                    self.gestures.update(now, left, right)
                                    self.gestures_armed = True
                            else:
                                self.gestures.update(now, left, right)
                                self.gestures.flush(now)
                cv2.putText(frame, f"GYRO: {self.gyro_status[:42]}",
                            (15, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.47, (255, 255, 255), 1, cv2.LINE_AA)
                if self.calibration_stage not in ("done", "error"):
                    prompts = {
                        "auto": "AUTO SETUP: LOOK AT CAMERA WITH EYES NATURALLY OPEN",
                        "open": "STEP 1/3: KEEP BOTH EYES OPEN",
                        "left": "STEP 2/3: GENTLY CLOSE LEFT EYE ONLY, THEN OPEN",
                        "right": "STEP 3/3: GENTLY CLOSE RIGHT EYE ONLY, THEN OPEN",
                    }
                    prompt = prompts[self.calibration_stage]
                    cv2.putText(frame, prompt, (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.62, (0, 220, 255), 2, cv2.LINE_AA)
                rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_image = Image.fromarray(rgb_preview).resize(
                    (640, 480), Image.Resampling.BILINEAR)
                image_tk = ImageTk.PhotoImage(preview_image)
                self.preview.configure(image=image_tk, text="")
                self.image_ref = image_tk
                if self.error_status.startswith("Camera/FaceLandmarker"):
                    self.error_status = ""
        except Exception as exc:
            self.error_status = f"Camera/FaceLandmarker error: {exc}"
            self.face_detected = False
            self.eye_tracking_status = "TRACKING ERROR — gestures stopped"
            self.eye_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
            self.last_ears = (None, None)
            self.last_eye_widths = (None, None)
            self.last_blink_scores = (None, None)
            self.eye_feature_history.clear()
            if self.calibration_stage == "auto":
                self.auto_baseline_samples = {
                    "ear": [[], []], "blink": [[], []]}
            self.gestures.lost_face()
            self.gestures_armed = False
            self._release_buttons(force=True)
        self._process_gyro(now)
        elapsed = max(1e-6, now - self.last_frame_time)
        self.fps = 0.9 * self.fps + 0.1 / elapsed if self.fps else 1.0 / elapsed
        self.last_frame_time = now
        self._update_status()
        elapsed_ms = (time.monotonic() - tick_started) * 1000
        delay_ms = max(1, round(TARGET_FRAME_INTERVAL_MS - elapsed_ms))
        self.root.after(delay_ms, self._tick)

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        for cleanup in (
            self.voice.stop, self.gyro.stop,
            lambda: self._release_buttons(force=True),
            self.cap.release, self.landmarker.close,
        ):
            try:
                cleanup()
            except Exception:
                pass
        if self._registered_hotkeys:
            user32 = ctypes.windll.user32
            user32.UnregisterHotKey(None, 1)
            user32.UnregisterHotKey(None, 2)
            self._registered_hotkeys = False
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        FaceGyroApp(root)
    except Exception as exc:
        try:
            messagebox.showerror("Face/gyro mouse startup error", str(exc), parent=root)
        except tk.TclError:
            pass
        try:
            root.destroy()
        except tk.TclError:
            pass
        print(f"Startup error: {exc}")
        raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
