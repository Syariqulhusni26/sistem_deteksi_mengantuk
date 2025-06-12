import streamlit as st
from playsound import playsound
import cv2
import dlib
import pygame
import numpy as np
import time
from scipy.spatial import distance
from datetime import datetime
import threading
import tempfile
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Inisialisasi session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Konstanta
LEFT_EYE_INDEX = list(range(36, 42))
RIGHT_EYE_INDEX = list(range(42, 48))
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME_THRESHOLD = 1.0
NOD_THRESHOLD = 8
ALARM_COOLDOWN = 5

# Variabel global untuk tracking
if 'eye_closed_start_time' not in st.session_state:
    st.session_state.eye_closed_start_time = None
if 'head_turn_start_time' not in st.session_state:
    st.session_state.head_turn_start_time = None
if 'yawn_start_time' not in st.session_state:
    st.session_state.yawn_start_time = None
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0

def load_models():
    """Load dlib models"""
    try:
        # Inisialisasi pygame untuk audio
        pygame.mixer.init()
        
        # Load detector wajah
        detector = dlib.get_frontal_face_detector()
        
        # Load predictor - coba beberapa nama file yang umum
        predictor_files = [
            "shape_predictor_68_face_landmarks.dat",
            "shape_predictor_68_face_landmarks copy.dat",
            "models/shape_predictor_68_face_landmarks.dat"
        ]
        
        predictor = None
        for file_path in predictor_files:
            if os.path.exists(file_path):
                try:
                    predictor = dlib.shape_predictor(file_path)
                    st.success(f"✅ Model predictor berhasil dimuat dari: {file_path}")
                    break
                except Exception as e:
                    st.warning(f"⚠️ Gagal memuat {file_path}: {e}")
                    continue
        
        if predictor is None:
            st.warning("⚠️ File model predictor tidak ditemukan. Sistem akan berjalan dalam mode deteksi wajah saja.")
            st.info("💡 Download model dari: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2")
        
        return detector, predictor
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

def calculate_ear(eye):
    """Menghitung Eye Aspect Ratio (EAR)"""
    # Jarak vertikal
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Jarak horizontal
    C = distance.euclidean(eye[0], eye[3])
    
    # Rumus EAR
    ear = (A + B) / (2.0 * C)
    return ear

def log_drowsiness_event(event_type):
    """Mencatat event kantuk ke session state"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'event': event_type
    }
    st.session_state.log_data.append(log_entry)

last_alarm_time = {"eye_closed": 0, "head_turn": 0, "yawning": 0}
ALARM_COOLDOWN = 5  # Detik

def play_audio_alert(alert_type):
    current_time = time.time()
    if current_time - last_alarm_time[alert_type] < ALARM_COOLDOWN:
        return  # Jangan mainkan alarm yang sama berulang kali dalam waktu singkat

    last_alarm_time[alert_type] = current_time

    def play_sound():
        if alert_type == "eye_closed":
            playsound("Alarm.mp3")
        elif alert_type == "head_turn":
            playsound("Alarm2.mp3")
        elif alert_type == "yawning":
            playsound("Alarm3.mp3")

    threading.Thread(target=play_sound, daemon=True).start()

def detect_drowsiness_features(frame, detector, predictor):
    """Deteksi fitur kantuk dari frame"""
    if predictor is None:
        return frame, {"status": "Model tidak tersedia"}
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    results = {
        "faces_detected": len(faces),
        "ear": 1.0,
        "mar": 0.0,
        "head_position": "center",
        "alerts": []
    }
    
    current_time = time.time()
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Ekstrak koordinat mata
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        
        # Hitung EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        results["ear"] = ear
        
        # Deteksi mata tertutup
        if ear < EAR_THRESHOLD:
            if st.session_state.eye_closed_start_time is None:
                st.session_state.eye_closed_start_time = current_time
            elif current_time - st.session_state.eye_closed_start_time >= CLOSED_EYE_TIME_THRESHOLD:
                if current_time - st.session_state.last_alarm_time > ALARM_COOLDOWN:
                    results["alerts"].append("Mata tertutup terlalu lama!")
                    log_drowsiness_event("Mata tertutup")
                    st.session_state.last_alarm_time = current_time
        else:
            st.session_state.eye_closed_start_time = None
        
        # Deteksi kepala menunduk
        nose = (landmarks.part(27).x, landmarks.part(27).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        head_tilt = abs(nose[1] - chin[1])
        
        if head_tilt < NOD_THRESHOLD:
            results["alerts"].append("Kepala menunduk!")
            log_drowsiness_event("Kepala menunduk")
        
        # Deteksi kepala menoleh
        nose_x = landmarks.part(30).x
        mid_face_x = (landmarks.part(36).x + landmarks.part(45).x) // 2
        
        if nose_x < mid_face_x - 20:
            results["head_position"] = "left"
        elif nose_x > mid_face_x + 20:
            results["head_position"] = "right"
        else:
            results["head_position"] = "center"
        
        if results["head_position"] in ["left", "right"]:
            if st.session_state.head_turn_start_time is None:
                st.session_state.head_turn_start_time = current_time
            elif current_time - st.session_state.head_turn_start_time > 2:
                if current_time - st.session_state.last_alarm_time > ALARM_COOLDOWN:
                    results["alerts"].append("Menoleh terlalu lama!")
                    log_drowsiness_event("Menoleh terlalu lama")
                    st.session_state.last_alarm_time = current_time
        else:
            st.session_state.head_turn_start_time = None
        
        # Deteksi menguap
        upper_lip = (landmarks.part(51).y + landmarks.part(62).y) / 2
        lower_lip = (landmarks.part(57).y + landmarks.part(66).y) / 2
        mouth_width = abs(landmarks.part(48).x - landmarks.part(54).x)
        
        mar = abs(upper_lip - lower_lip) / mouth_width if mouth_width > 0 else 0
        results["mar"] = mar
        
        if mar > 0.5:
            if st.session_state.yawn_start_time is None:
                st.session_state.yawn_start_time = current_time
            elif current_time - st.session_state.yawn_start_time > 1.3:
                if current_time - st.session_state.last_alarm_time > ALARM_COOLDOWN:
                    results["alerts"].append("Menguap terdeteksi!")
                    log_drowsiness_event("Menguap")
                    st.session_state.last_alarm_time = current_time
        else:
            st.session_state.yawn_start_time = None
        
        # Gambar deteksi pada frame
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Gambar titik mata
        for (ex, ey) in left_eye + right_eye:
            cv2.circle(frame, (ex, ey), 2, (255, 255, 255), -1)
    
    return frame, results

def main():
    st.title("🚗 Driver Drowsiness Detection System")
    st.markdown("---")
    
    # Sidebar untuk kontrol
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Pengaturan threshold
        ear_threshold = st.slider("EAR Threshold", 0.1, 0.5, EAR_THRESHOLD, 0.01)
        time_threshold = st.slider("Waktu Mata Tertutup (detik)", 0.5, 3.0, CLOSED_EYE_TIME_THRESHOLD, 0.1)
        
        st.markdown("---")
        
        # Status sistem
        st.header("📊 Status Sistem")
        if st.session_state.detector is None:
            st.error("Model belum dimuat")
            if st.button("Load Models"):
                detector, predictor = load_models()
                st.session_state.detector = detector
                st.session_state.predictor = predictor
                if detector:
                    st.success("Model berhasil dimuat!")
                    st.rerun()
        else:
            st.success("Model siap digunakan")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Video Stream")
        
        # Placeholder untuk video
        video_placeholder = st.empty()
        
        # Kontrol kamera
        col_start, col_stop = st.columns(2)
        with col_start:
            start_camera = st.button("🎥 Start Camera", disabled=st.session_state.camera_active)
        with col_stop:
            stop_camera = st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_active)
        
        # Inisialisasi variabel untuk real-time data
        if 'current_ear' not in st.session_state:
            st.session_state.current_ear = 0.30
        if 'current_head_pos' not in st.session_state:
            st.session_state.current_head_pos = "Center"
        if 'current_mar' not in st.session_state:
            st.session_state.current_mar = 0.20
        
        # Kamera real-time dengan OpenCV
        if start_camera and st.session_state.detector:
            st.session_state.camera_active = True
            
            # Inisialisasi kamera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Tidak dapat mengakses kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
            else:
                st.success("✅ Kamera berhasil terhubung!")
                
                # Loop untuk video streaming
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Gagal membaca frame dari kamera")
                        break
                    
                    # Flip frame horizontal agar seperti cermin
                    frame = cv2.flip(frame, 1)
                    
                    # Deteksi wajah dan fitur kantuk
                    if st.session_state.predictor is not None:
                        processed_frame, results = detect_drowsiness_features(
                            frame, st.session_state.detector, st.session_state.predictor
                        )
                        
                        for alert in results["alerts"]:
                            if "Mata tertutup" in alert:
                                play_audio_alert("eye_closed")
                            elif "Menoleh" in alert:
                                play_audio_alert("head_turn")
                            elif "Menguap" in alert:
                                play_audio_alert("yawning")

                        
                        # Update real-time data
                        st.session_state.current_ear = results.get("ear", 0.30)
                        st.session_state.current_head_pos = results.get("head_position", "Center")
                        st.session_state.current_mar = results.get("mar", 0.20)
                        
                        # Tampilkan alerts
                        for alert in results.get("alerts", []):
                            st.warning(f"🚨 {alert}")
                    else:
                        processed_frame = frame
                        # Gambar kotak deteksi wajah sederhana tanpa landmarks
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = st.session_state.detector(gray)
                        
                        for face in faces:
                            x, y, w, h = face.left(), face.top(), face.width(), face.height()
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(processed_frame, "Face Detected", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Konversi frame ke RGB untuk Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Tampilkan frame di Streamlit
                    video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Small delay to prevent too much CPU usage
                    time.sleep(0.03)  # ~30 FPS
                
                # Release kamera ketika selesai
                cap.release()
                
        if stop_camera:
            st.session_state.camera_active = False
            video_placeholder.empty()
            st.info("⏹️ Kamera dihentikan")
    
    
    
    with col2:
        st.header("📈 Monitoring")
        
        # Metrics real-time
        if st.session_state.camera_active:
            # Update metrics dengan data real-time
            ear_value = st.session_state.current_ear
            head_pos = st.session_state.current_head_pos
            mar_value = st.session_state.current_mar
            
            # Tentukan status berdasarkan nilai
            ear_status = "⚠️ Mengantuk" if ear_value < ear_threshold else "✅ Normal"
            head_status = "✅ Baik" if head_pos == "Center" else f"⚠️ {head_pos}"
            mar_status = "⚠️ Menguap" if mar_value > 0.5 else "✅ Normal"
            
            st.metric("EAR Value", f"{ear_value:.2f}", ear_status)
            st.metric("Posisi Kepala", head_pos, head_status)
            st.metric("MAR Value", f"{mar_value:.2f}", mar_status)
        else:
            st.metric("EAR Value", "0.30", "Standby")
            st.metric("Posisi Kepala", "Center", "Standby")
            st.metric("MAR Value", "0.20", "Standby")
        
        # Alert panel
        st.subheader("🚨 Alert Status")
        if st.session_state.camera_active:
            current_time = datetime.now().strftime('%H:%M:%S')
            if st.session_state.current_ear < ear_threshold:
                st.error(f"🚨 [{current_time}] MATA TERTUTUP!")
            elif st.session_state.current_head_pos != "Center":
                st.warning(f"⚠️ [{current_time}] KEPALA MENOLEH!")
            elif st.session_state.current_mar > 0.5:
                st.warning(f"⚠️ [{current_time}] MENGUAP TERDETEKSI!")
            else:
                st.success(f"✅ [{current_time}] Status Normal")
        else:
            st.info("📱 Sistem dalam mode standby")
        
        # Log events
        st.subheader("📝 Log Events")
        if st.session_state.log_data:
            # Tampilkan log dalam container yang bisa di-scroll
            log_container = st.container()
            with log_container:
                for log in st.session_state.log_data[-10:]:  # Show last 10 events
                    st.text(f"{log['timestamp']}: {log['event']}")
        else:
            st.text("Belum ada event tercatat")
        
        if st.button("🗑️ Clear Log"):
            st.session_state.log_data = []
            st.rerun()
    
    # Informasi sistem
    st.markdown("---")
    st.header("ℹ️ Informasi Sistem")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total Events", len(st.session_state.log_data))
    with col_info2:
        st.metric("EAR Threshold", f"{ear_threshold:.2f}")
    with col_info3:
        st.metric("Time Threshold", f"{time_threshold:.1f}s")
    
    # Instructions
    with st.expander("📖 Cara Penggunaan"):
        st.markdown("""
        1. **Load Models**: Klik tombol "Load Models" di sidebar
        2. **Start Camera**: Klik "Start Camera" untuk memulai deteksi
        3. **Monitor**: Pantau metrics dan alerts di panel kanan
        4. **Alerts**: Sistem akan memberikan peringatan jika:
           - Mata tertutup terlalu lama (EAR < threshold)
           - Kepala menunduk (mengantuk)
           - Kepala menoleh terlalu lama
           - Terdeteksi menguap
        
        **Catatan**: Untuk implementasi penuh, diperlukan:
        - File model `shape_predictor_68_face_landmarks.dat`
        - File audio alarm (MP3)
        - Server dengan akses kamera
        """)

if __name__ == "__main__":
    main()

