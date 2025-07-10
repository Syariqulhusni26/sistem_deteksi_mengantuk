import os
import time # untuk timer waktu
import cv2
import dlib
import winsound
import pygame
import math
from scipy.spatial import distance

# Load Model file Landmark
model_path = "shape_predictor_68_face_landmarks copy.dat"

# Load model deteksi wajah dari dlib
detector = dlib.get_frontal_face_detector()

# Load Model predictor untuk mendeteksi landmark wajah
predictor = dlib.shape_predictor(model_path)

# Membuat landmark mata kiri dan kanan
LEFT_EYE_INDEX = list(range(36, 42))
RIGHT_EYE_INDEX = list(range(42, 48))

# Variabel untuk membuat ambang batas EAR dibawah 0,25
EAR_THRESHOLD = 0.25

# Variabel untuk waktu agar bunyika alarm ketika mata tertutup
CLOSED_EYE_TIME_THRESHOLD = 1.0

# Variabel untuk mencatat LOG tidur
LOG_FILE = "drowsiness_log.txt"

# Variabel untuk melihat ambang batas mendeteksi kepala menunduk
NOD_THRESHOLD = 8

# Inisialisasi pygame mixer untuk suara dari mp3
pygame.mixer.init()

# Variabel global untuk tracking waktu mata tertutup
eye_closed_start_time = None

# Variabel global untuk tracking kepala menoleh kekiri dan kekanan
head_turn_start_time = None

# Variabel global untuk tracking supir menguap
yawn_start_time = None

# Variabel penghitung deteksi
count_drowsy = 0
count_eye_closed = 0
count_head_turn = 0
count_head_right_left = 0
count_yawn = 0

# Variabel untuk menyimpan waktu alarm terakhir berbunyi
last_alarm_time = 0  # Menyimpan waktu terakhir alarm berbunyi
alarm_cooldown = 5   # Jeda alarm dalam detik


# Fungsi untuk memutar suara alarm
def play_alarm():
    # pygame.mixer.music.load("Alarm.mp3")  # Sesuaikan dengan nama file MP3 kamu
    pygame.mixer.music.load("Alarm.mp3")
    pygame.mixer.music.play()
    
def play_alarm1():
    # pygame.mixer.music.load("Alarm.mp3")  # Sesuaikan dengan nama file MP3 kamu
    pygame.mixer.music.load("Alarm1.mp3")
    pygame.mixer.music.play()
    
def play_alarm2():
    # pygame.mixer.music.load("Alarm.mp3")  # Sesuaikan dengan nama file MP3 kamu
    pygame.mixer.music.load("Alarm2.mp3")
    pygame.mixer.music.play()
    
def play_alarm3():
    # pygame.mixer.music.load("Alarm.mp3")  # Sesuaikan dengan nama file MP3 kamu
    pygame.mixer.music.load("Alarm3.mp3")
    pygame.mixer.music.play()    
    
# Fungsi untuk mencatat waktu tidur ke dalam file lgo
def log_drowsiness():
    with open(LOG_FILE, "a") as log :
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Supir mengantuk!\n")
        print("[LOG] Supir mengantuk, dicatat dalam log.")

# Fungsi input kamera video
def get_video_stream(source=0):
    video_strean = cv2.VideoCapture(source)
    return video_strean
    
# Fungsi untuk menghitung aspek rasio mata dari landmarks Wajah
def calculate_ear(eye):
    
    # Variabel untuk menghitung jarak vertikal
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Variabel utnuk menghitung jarak horizontal dari kedua mata
    C = distance.euclidean(eye[0], eye[3])
    
    # Rumus mencari EAR
    ear = (A + B) / (2.0 * C)
    return ear


# Fungsi untuk mendeteksi wajah menggunakan Dlib
def detect_faces_and_eye(frame, detector, predictor):
    
    global eye_closed_start_time # Variabel Global waktu mata tertutup
    global head_turn_start_time # Variabel Global waktu kepala turun
    global last_alarm_time  # Variabel Global waktu terakhir
    global yawn_start_time  # Variabel Global waktu menguap
        
    # global counter juga di sini
    global count_drowsy, count_eye_closed, count_head_turn, count_yawn, count_head_right_left

    # Variabel untuk mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Variabel untuk mendeteksi wajah dalam frame
    faces = detector(gray)
    left_eye, right_eye = None, None
    ear = 1.0
    
    # Variabel untuk mendeteksi  supir menguap
    mar = 0.0
    
    for face in faces:
        # Variabel untuk mendapatkan landmark wajah
        landmark = predictor(gray, face)
        
        #variabel untuk membuat titik titik pada landmark mata kiri dan kanan
        left_eye = [(landmark.part(n).x, landmark.part(n).y) for n in range(36, 42)]
        right_eye = [(landmark.part(n).x, landmark.part(n).y) for n in range(42, 48)]
        
        # Variabel untuk menghitung nilai EAR pada kedua mata
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        # Variabel untuk Rata rata EAR pada kedua mata
        ear = (left_ear + right_ear) / 2.0
        
        # 1. Fitur untuk deteksi kepala menunduk
        head_down = False

        nose = (landmark.part(27).x, landmark.part(27).y)
        chin = (landmark.part(8).x, landmark.part(8).y)

        # Hitung jarak vertikal antara hidung dan dagu
        head_tilt = abs(nose[1] - chin[1])

        # # Tambahkan visual garis dari hidung ke dagu
        # cv2.line(frame, nose, chin, (0, 0, 255), 2)

        # # Debug nilai head_tilt
        # print(f"[DEBUG] Head tilt (hidung ke dagu): {head_tilt}")

        # Threshold dinamis (misal wajah dekat ke kamera, head_tilt > 50)
        NOD_THRESHOLD = 115  # Bisa disesuaikan dari hasil pengamatan

        current_time = time.time()
        if head_tilt < NOD_THRESHOLD:
            if current_time - last_alarm_time > alarm_cooldown:
                count_drowsy += 1
                count_head_turn += 1
                print("[ALERT] Kepala menunduk! Supir mungkin mengantuk.")
                winsound.Beep(1000, 500)
                play_alarm()
                log_drowsiness()
                last_alarm_time = current_time
                head_down = True
            
        #2.  Fitur untuk deteksi mata supir tertutup 
        # Perulangan untuk mengecek apakah mata tertutup atau tidak
        if ear < EAR_THRESHOLD:
            if eye_closed_start_time is None :
                eye_closed_start_time = time.time() # Memulai waktu mundur
            elif time.time() - eye_closed_start_time >= CLOSED_EYE_TIME_THRESHOLD:
                current_time = time.time()
                if current_time - last_alarm_time > alarm_cooldown:
                    count_drowsy += 1
                    count_eye_closed += 1
                    print("[ALERT] Jangan Mengantuk !!!")
                    winsound.Beep(5000, 900)
                    play_alarm1()  # Bunyi alarm
                    log_drowsiness() # Mencatat Log
                    last_alarm_time = current_time
        else:
            eye_closed_start_time = None # Timer direset ketika mata terbuka
        
        #3.  Fitur untuk deteksi supir menoleh kekanan atau kekiri
        # Variabel untuk mengambil kordinat landmark hidung
        nose_x = landmark.part(30).x
        
        #Variabel untuk mengambil kordinat tengah wajah
        mid_face_x = (landmark.part(36).x + landmark.part(45).x) // 2
        
        # buat if else jika hidung terlalu ke kanan atau kekiri dari tengah wajah
        
        if nose_x < mid_face_x - 20 : # Ketika kepala menoleh kekiri
            head_direction = "left"
        elif nose_x > mid_face_x + 20: # Ketika kepala menoleh ke kanan
            head_direction = "right"
        else:
            head_direction = "center"
            
        if head_direction in["left", "right"]:
            if head_turn_start_time is None :
                head_turn_start_time = time.time() # Fungsi untuk memulai waktu
            elif time.time() - head_turn_start_time > 2:
                current_time = time.time()
                if current_time - last_alarm_time > alarm_cooldown:  # Cek jeda alarm
                    print("[ALERT] Tetap Fokus Saat Berkendara!!! Jangan Menoleh Terlalu Lama")
                    count_drowsy += 1
                    count_head_right_left += 1
                    winsound.Beep(1500, 300)
                    play_alarm2()
                    log_drowsiness() # Mencatat Log
                    last_alarm_time = current_time
        else:
            head_turn_start_time = None
        
        #4.  Fitur untuk deteksi supir menguap
        # Variabel untuk mengambil kordinat mulut
        upper_lip = (landmark.part(51).y + landmark.part(62).y) / 2
        lower_lip = (landmark.part(57).y + landmark.part(66).y) / 2
        mouth_with = abs(landmark.part(48).x - landmark.part(54).x) 
        
        # Variabel untuk menghitung MAR (Mouth Aspect Ratio)
        mar = abs(upper_lip - lower_lip) / mouth_with
        
        # Buat If Else jika mulut terbuka, maka beri peringatan
        
        if mar > 0.5:
            if yawn_start_time is None:
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time > 1.3:
                current_time = time.time()
                if current_time - last_alarm_time > alarm_cooldown :
                    print("[ALERT] Supir Menguap!!! Istirahat Sejenak")
                    count_drowsy += 1
                    count_yawn += 1
                    winsound.Beep(1700, 400)
                    play_alarm3()
                    log_drowsiness()
                    last_alarm_time = current_time
        else:
            yawn_start_time = None
        
    return faces, left_eye, right_eye, ear, mar
    
# Mengatur Resolusi kamera
def get_video_stream():
    cap = cv2.VideoCapture(0)
    
    # Set resolusi ke 1280x720 (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

SMOOTHING_FACTOR = 0.3
prev_face_coords = None
prev_eye_coords = None

def main():
    global prev_face_coords, prev_eye_coords
    global count_drowsy, count_eye_closed, count_head_turn, count_head_right_left, count_yawn

    video_stream = get_video_stream()

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break  # Jika kamera tidak terbuka, keluar dari loop

        # Deteksi wajah dan fitur
        faces, left_eye, right_eye, ear, mar = detect_faces_and_eye(frame, detector, predictor)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # ➤ Smooth kotak wajah
            if prev_face_coords is None:
                prev_face_coords = (x, y, w, h)
            else:
                x = int(prev_face_coords[0] * (1 - SMOOTHING_FACTOR) + x * SMOOTHING_FACTOR)
                y = int(prev_face_coords[1] * (1 - SMOOTHING_FACTOR) + y * SMOOTHING_FACTOR)
                w = int(prev_face_coords[2] * (1 - SMOOTHING_FACTOR) + w * SMOOTHING_FACTOR)
                h = int(prev_face_coords[3] * (1 - SMOOTHING_FACTOR) + h * SMOOTHING_FACTOR)
                prev_face_coords = (x, y, w, h)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ➤ Gambar titik-titik mata dengan smoothing
            if left_eye and right_eye:
                all_eyes = left_eye + right_eye
                if prev_eye_coords is None:
                    prev_eye_coords = all_eyes
                else:
                    smoothed_eyes = []
                    for i in range(len(all_eyes)):
                        px, py = prev_eye_coords[i]
                        cx, cy = all_eyes[i]
                        sx = int(px * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
                        sy = int(py * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)
                        smoothed_eyes.append((sx, sy))
                    prev_eye_coords = smoothed_eyes

                    for (ex, ey) in smoothed_eyes:
                        cv2.circle(frame, (ex, ey), 2, (255, 255, 255), -1)

        # Menampilkan info counter di frame
        info_text = [
            f"Drowsy Count     : {count_drowsy}",
            f"Mata Tertutup    : {count_eye_closed}",
            f"Kepala Menunduk  : {count_head_turn}",
            f"Kepala Menoleh   : {count_head_right_left}",
            f"Menguap          : {count_yawn}"
        ]

        # Tentukan posisi dan ukuran box info
        # Ukuran font dan spasi
        font_scale = 0.5
        line_spacing = 20
        text_color = (0, 255, 255)
        font = cv2.FONT_HERSHEY_DUPLEX

        # Dapatkan ukuran frame
        frame_height, frame_width = frame.shape[:2]

        # Hitung tinggi box info
        box_height = line_spacing * len(info_text) + 20
        box_width = 270  # sesuaikan lebar box

        # Posisi pojok kanan bawah
        x = frame_width - box_width - 10
        y = frame_height - box_height - 10

        # Gambar background semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 5), (x + box_width, y + box_height), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Tampilkan teks
        for i, text in enumerate(info_text):
            text_y = y + 20 + i * line_spacing
            cv2.putText(frame, text, (x + 10, text_y), font,
                        font_scale, text_color, 1, cv2.LINE_AA)



        cv2.imshow("Deteksi Mata & Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Tekan 'q' untuk keluar
            break

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()