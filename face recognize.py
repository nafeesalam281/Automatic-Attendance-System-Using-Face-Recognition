import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load models
print("[INFO] Loading models...")
embedder = FaceNet()
model = YOLO("yolov8n-face.pt")

# Load embeddings and labels
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# Load student records
students_df = pd.read_csv("Student Record.csv")
students_df["Name"] = students_df["Name"].str.strip()

# Mapping name → (Reg.No, Mob.No)
name_to_info = {
    row["Name"].strip().lower(): (row["Reg.No"], str(row["Mob.No"]))
    for _, row in students_df.iterrows()
}

# Encode dataset names
label_encoder = LabelEncoder()
dataset_names = sorted(list(set([name.split("_")[0].strip().lower() for name in os.listdir("dataset")])))
label_encoder.fit(dataset_names)

# Start webcam
cap = cv2.VideoCapture(0)
attendance = {}
print("[INFO] Starting webcam...")

frame_count = 0
MAX_FRAMES = 100

# Function to clean mobile number
def clean_number(number):
    number = str(number).strip().replace("+91", "").replace("-", "").replace(" ", "")
    return number if number.isdigit() and len(number) == 10 else None

# Function to send SMS using Fast2SMS
def send_sms(name, number, date, status):
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {
        'authorization': "Liv24y3WyaXKeZsjBWdoJCB85riNhy5Hyh0ajkiSLzI7zkU30ANchSDdkQ8g",  # Replace with your Fast2SMS API Key
        'Content-Type': 'application/json'
    }

    message = f"{name}, you have been marked {status} on {date}."
    payload = {
        "route": "q",
        "sender_id": "FSTSMS",
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": number
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Face recognition and attendance
while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            face = cv2.resize(face, (160, 160))
            face_array = np.expand_dims(face, axis=0)
            face_embed = embedder.embeddings(face_array)[0]

            sims = cosine_similarity([face_embed], embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            name = label_encoder.inverse_transform([labels[best_idx]])[0] if best_score > 0.6 else "Unknown"
            name = name.strip().lower()

            if name != "unknown" and name not in attendance:
                now = datetime.now()
                attendance[name] = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[✓] Marked Present: {name}")

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)
    cv2.waitKey(1)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Record attendance
date_today = datetime.now().strftime("%Y-%m-%d")
attendance_list = []

print("\n[INFO] Generating attendance sheet...")

for _, row in students_df.iterrows():
    name_raw = row["Name"].strip()
    name_key = name_raw.lower()
    reg_no = row["Reg.No"]

    if name_key in attendance:
        time = attendance[name_key].split()[1]
        status = "Present"
    else:
        time = "-"
        status = "Absent"

    attendance_list.append([name_raw, reg_no, date_today, time, status])

# Save CSV
daily_file = f"attendance_{date_today}.csv"
daily_df = pd.DataFrame(attendance_list, columns=["Name", "Reg_No", "Date", "Time", "Status"])
daily_df.to_csv(daily_file, index=False)
print(f"[✓] Daily attendance saved → {daily_file}")

# Send SMS to all students with status
print("\n[INFO] Sending SMS to all students...")

for _, row in students_df.iterrows():
    name_raw = row["Name"].strip()
    name_key = name_raw.lower()
    raw_no = str(row["Mob.No"])
    mob_no = clean_number(raw_no)

    if name_key in attendance:
        status = "Present"
    else:
        status = "Absent"

    if mob_no:
        try:
            result = send_sms(name_raw, mob_no, date_today, status)
            print(f"[INFO] Sent SMS to {name_raw} ({status}): {result}")
        except Exception as e:
            print(f"[ERROR] Failed to send SMS to {name_raw}: {e}")
    else:
        print(f"[WARNING] Invalid mobile number for {name_raw}: {raw_no}")