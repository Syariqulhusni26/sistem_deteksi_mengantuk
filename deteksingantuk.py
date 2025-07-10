import gradio as gr
import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance
from datetime import datetime
import os

# Konstanta dan threshold
LEFT_EYE_INDEX = list(range(36, 42))
RIGHT_EYE_INDEX = list(range(42, 48))
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME_THRESHOLD = 1.0
ALARM_COOLDOWN = 5

# Variabel global
eye_closed_start_time = None
head_turn_start_time = None
yawn_start_time = None
last_alarm_time = 0
log_data = []

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def log_drowsiness_event(event_type):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data.append({"timestamp": timestamp, "event": event_type})

def detect_drowsiness(frame):
    global eye_closed_start_time, head_turn_start_time, yawn_start_time, last_alarm_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_time = time.time()
    alerts = []

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_INDEX]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_INDEX]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
            elif current_time - eye_closed_start_time >= CLOSED_EYE_TIME_THRESHOLD:
                if current_time - last_alarm_time > ALARM_COOLDOWN:
                    alerts.append("Mata tertutup terlalu lama!")
                    log_drowsiness_event("Mata tertutup")
                    last_alarm_time = current_time
        else:
            eye_closed_start_time = None

        # Deteksi kepala menoleh
        nose_x = landmarks.part(30).x
        mid_face_x = (landmarks.part(36).x + landmarks.part(45).x) // 2
        head_position = "center"
        if nose_x < mid_face_x - 20:
            head_position = "left"
        elif nose_x > mid_face_x + 20:
            head_position = "right"

        if head_position in ["left", "right"]:
            if head_turn_start_time is None:
                head_turn_start_time = current_time
            elif current_time - head_turn_start_time > 2:
                if current_time - last_alarm_time > ALARM_COOLDOWN:
                    alerts.append("Menoleh terlalu lama!")
                    log_drowsiness_event("Menoleh terlalu lama")
                    last_alarm_time = current_time
        else:
            head_turn_start_time = None

        # Deteksi menguap
        upper_lip = (landmarks.part(51).y + landmarks.part(62).y) / 2
        lower_lip = (landmarks.part(57).y + landmarks.part(66).y) / 2
        mouth_width = abs(landmarks.part(48).x - landmarks.part(54).x)
        mar = abs(upper_lip - lower_lip) / mouth_width if mouth_width > 0 else 0

        if mar > 0.5:
            if yawn_start_time is None:
                yawn_start_time = current_time
            elif current_time - yawn_start_time > 1.3:
                if current_time - last_alarm_time > ALARM_COOLDOWN:
                    alerts.append("Menguap terdeteksi!")
                    log_drowsiness_event("Menguap")
                    last_alarm_time = current_time
        else:
            yawn_start_time = None

        # Gambar wajah & mata
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (ex, ey) in left_eye + right_eye:
            cv2.circle(frame, (ex, ey), 2, (255, 255, 255), -1)

    for i, alert in enumerate(alerts):
        cv2.putText(frame, alert, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, "\n".join(alerts) if alerts else "Tidak ada tanda ngantuk"

# Fungsi utama untuk Gradio
def process_frame(img):
    if img is None:
        return None, "Tidak ada gambar dari kamera."

    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed_frame, status = detect_drowsiness(frame)
    return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), status

# Load model Dlib
def load_models():
    file = "shape_predictor_68_face_landmarks copy.dat"
    if os.path.exists(file):
        return dlib.get_frontal_face_detector(), dlib.shape_predictor(file)
    raise FileNotFoundError("Model shape_predictor_68_face_landmarks.dat tidak ditemukan.")

# Load model global
try:
    detector, predictor = load_models()
except Exception as e:
    raise SystemExit(f"Gagal memuat model: {e}")

# Antarmuka Gradio
interface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Arahkan Kamera ke Wajah"
    ),
    outputs=[
        gr.Image(label="Hasil Deteksi"),
        gr.Textbox(label="Status Deteksi")
    ],
    live=True
)

interface.launch()  