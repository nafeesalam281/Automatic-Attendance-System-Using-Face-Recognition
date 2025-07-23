import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

name = input("Enter your name: ").strip()
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        face = frame[y1:y2, x1:x2]


        if face.size > 0:
            count += 1
            face = cv2.resize(face, (160, 160))
            path = os.path.join(dataset_dir, f"{name}_{count}.jpg")
            cv2.imwrite(path, face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} {count}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[✓] Collected {count} images for {name}")
