import sys
from pathlib import Path
import cv2
import numpy as np
import os
import threading
import mysql.connector
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QLineEdit, QFileDialog, QMessageBox, QComboBox, QDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
import face_recognition
import time
from playsound import playsound

# === Konstanta ===
YOLO_PATH = "yolov8n-face.pt"
SOUND_PATH = Path("sound/absen.mp3")
DB_IMAGE_PATH = Path("db_image")
ABSEN_IMAGE_PATH = Path("absen")
SHUTTER_SOUND_PATH = Path("sound/shutter.mp3")
REGISTRATION_SOUND_PATH = Path("sound/registration.mp3")

# === Global State ===
known_ids, known_names, known_encodings = [], [], []
last_logged_time = {}
cap = None

# === Inisialisasi Model ===
yolo_model = YOLO(YOLO_PATH).to("cuda")

# === Database ===
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        port=3306,
        database="attedance"
    )

def load_known_faces():
    global known_ids, known_names, known_encodings
    known_ids.clear()
    known_names.clear()
    known_encodings.clear()

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, face_encoding FROM Users")
    data = cursor.fetchall()
    conn.close()

    for user_id, name, encoding in data:
        if encoding:
            known_ids.append(user_id)
            known_names.append(name)
            known_encodings.append(np.frombuffer(encoding, dtype=np.float64))


def record_attendance(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    now = datetime.now()
    current_date = now.date()

    # Tentukan tipe absensi berdasarkan jam
    attendance_type = "masuk" if now.hour < 12 else "pulang"

    # Cek apakah user sudah absen dengan tipe yang sama hari ini
    cursor.execute("""
        SELECT COUNT(*) FROM AttendanceLog
        WHERE user_id = %s AND DATE(time) = %s AND type = %s
    """, (user_id, current_date, attendance_type))
    already_absent = cursor.fetchone()[0]

    # Jika belum absen, catat absensi
    if already_absent == 0:
        cursor.execute("""
            INSERT INTO AttendanceLog (user_id, time, type)
            VALUES (%s, %s, %s)
        """, (user_id, now, attendance_type))
        conn.commit()

        # Mainkan suara jika file tersedia
        if SOUND_PATH.exists():
            threading.Thread(target=playsound, args=(str(SOUND_PATH),), daemon=True).start()

    conn.close()




# === GUI ===
class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Absensi Wajah Kantor (YOLOv8 + PyQt5)")
        self.image_label = QLabel()
        self.info_label = QLabel("Status: Siap")
        self.fps_label = QLabel("FPS: 0")
        self.start_button = QPushButton("Mulai Kamera")
        self.stop_button = QPushButton("Berhenti Kamera")
        self.register_button = QPushButton("Registrasi Wajah")
        self.export_button = QPushButton("Export Log Absensi")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.register_button.clicked.connect(self.open_register_form)
        self.export_button.clicked.connect(self.export_log)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.info_label)

        btns = QHBoxLayout()
        btns.addWidget(self.start_button)
        btns.addWidget(self.stop_button)
        btns.addWidget(self.register_button)
        btns.addWidget(self.export_button)
        layout.addLayout(btns)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.last_frame_time = time.time()

    def start_camera(self):
        global cap
        cap = cv2.VideoCapture(0)
        self.timer.start(10)

    def stop_camera(self):
        global cap
        if cap:
            cap.release()
        self.timer.stop()
        self.image_label.clear()
        self.fps_label.setText("FPS: 0")
        self.info_label.setText("Status: Kamera berhenti")

    def update_frame(self):
        global cap
        ret, frame = cap.read()
        if not ret:
            return

        results = yolo_model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        face_locations = [(int(y1), int(x2), int(y2), int(x1)) for x1, y1, x2, y2 in boxes]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    user_id = known_ids[best_match]
                    name = known_names[best_match]
                    now = time.time()
                    if user_id not in last_logged_time or now - last_logged_time[user_id] > 10:
                        record_attendance(user_id)
                        last_logged_time[user_id] = now
                        ABSEN_IMAGE_PATH.mkdir(exist_ok=True)
                        filename = f"{name}_{datetime.now().strftime('%H-%M-%S')}.jpg"
                        cv2.imwrite(str(ABSEN_IMAGE_PATH / filename), frame)
                        self.info_label.setText(f"{name} absen")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps = 1 / (time.time() - self.last_frame_time + 1e-5)
        self.last_frame_time = time.time()
        self.fps_label.setText(f"FPS: {fps:.2f}")

        img = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def open_register_form(self):
        reg = RegisterForm()
        reg.exec_()

    def export_log(self):
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Users.name, Users.address, Users.department, AttendanceLog.time, AttendanceLog.type
            FROM AttendanceLog
            JOIN Users ON Users.id = AttendanceLog.user_id
            ORDER BY AttendanceLog.time DESC
        """)
        data = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(data, columns=["Nama", "Alamat", "Departemen", "Waktu", "Tipe"])
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan sebagai Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Berhasil", "Log absensi berhasil diekspor.")


# === Registrasi Wajah Multi-Angle Manual ===
class RegisterForm(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Registrasi Wajah Baru")
        self.setFixedSize(400, 500)

        self.cam = cv2.VideoCapture(0)
        self.label = QLabel()
        self.name_input = QLineEdit()
        self.addr_input = QLineEdit()
        self.dept_input = QComboBox()
        self.dept_input.addItems(["HRD", "Keuangan", "Teknik", "Lainnya"])
        self.capture_btn = QPushButton("Ambil Foto")
        self.capture_btn.clicked.connect(self.capture)

        # Instruksi sudut wajah
        self.directions = ["depan", "kiri", "kanan", "atas", "bawah"]
        self.current_dir = 0
        self.photos_per_dir = 2
        self.photo_count = 0

        self.instruction_label = QLabel(f"Arahkan wajah ke: {self.directions[self.current_dir]}")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Nama:"))
        layout.addWidget(self.name_input)
        layout.addWidget(QLabel("Alamat:"))
        layout.addWidget(self.addr_input)
        layout.addWidget(QLabel("Departemen:"))
        layout.addWidget(self.dept_input)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.label)
        layout.addWidget(self.capture_btn)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.show_frame)
        self.timer.start(30)

    def show_frame(self):
        ret, frame = self.cam.read()
        if ret:
            img = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         frame.shape[1], frame.shape[0],
                         frame.strides[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))
            self.frame = frame
    def capture(self):
        name = self.name_input.text().strip()
        addr = self.addr_input.text().strip()
        dept = self.dept_input.currentText()
        frame = self.frame

        if not name or not addr:
            QMessageBox.warning(self, "Invalid", "Isi semua data.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        if not face_encodings:
            QMessageBox.warning(self, "Gagal", "Tidak ada wajah terdeteksi.")
            return

        folder = DB_IMAGE_PATH / name.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        conn = connect_db()
        cursor = conn.cursor()
        for i, (encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
            direction = self.directions[self.current_dir]
            filename = f"{name}_{direction}_{self.photo_count}.jpg"
            filepath = folder / filename
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(str(filepath), face_img)

            # 🔊 Suara shutter saat capture
            if SHUTTER_SOUND_PATH.exists():
                threading.Thread(target=playsound, args=(str(SHUTTER_SOUND_PATH),), daemon=True).start()

            cursor.execute("""
                INSERT INTO Users (name, address, department, face_image, face_encoding)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, addr, dept, str(filepath), encoding.tobytes()))
            conn.commit()

        self.photo_count += 1

        if self.photo_count >= self.photos_per_dir:
            self.photo_count = 0
            self.current_dir += 1
            if self.current_dir >= len(self.directions):
                self.timer.stop()
                self.cam.release()

                # 🔊 Suara berhasil registrasi
                if REGISTRATION_SOUND_PATH.exists():
                    threading.Thread(target=playsound, args=(str(REGISTRATION_SOUND_PATH),), daemon=True).start()

                QMessageBox.information(self, "Sukses", "Registrasi selesai!")
                load_known_faces()
                self.close()
                return

        # Update instruksi sudut
        self.instruction_label.setText(f"Arahkan wajah ke: {self.directions[self.current_dir]}")

if __name__ == "__main__":
    load_known_faces()
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
