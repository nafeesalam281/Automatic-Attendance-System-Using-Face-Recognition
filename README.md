# Face Recognition Attendance System

This project is an automated attendance system using Python, OpenCV, YOLOv8 for face detection, and FaceNet for recognition.  
It automatically marks attendance, sends SMS alerts to absentees And present students using Fast2SMS API, and generates weekly reports.  

## Features
- Real-time face detection and recognition
- Automated attendance logging in CSV
- SMS alerts for absentees
- Weekly attendance percentage report
- Web integration with responsive UI

## Technologies Used
- Python
- OpenCV
- YOLOv8-face
- FaceNet
- Fast2SMS API
- Pandas

## Accuracy
- Detection Accuracy: 98.7%
- Recognition Accuracy: 95.2%

## Requirements

This project uses the following Python libraries:

- **opencv-python (cv2)** → For video capture, face preprocessing, and image operations  
- **ultralytics (YOLOv8)** → For real-time face detection  
- **tensorflow / keras** → For FaceNet embeddings (face recognition)  
- **scikit-learn** → For embedding comparison and classification  
- **numpy** → For numerical operations and array handling  
- **pandas** → For attendance CSV logging and weekly percentage calculation  
   **requests** → For sending absentee SMS alerts via Fast2SMS API  
- **datetime ** → For timestamps in attendance records  
- **os ** → For file handling and dataset management  

### Installation
You can install all required libraries using pip:

```bash
pip install opencv-python ultralytics tensorflow keras scikit-learn numpy pandas  requests


## Models Used

- **YOLOv8n-face** → Used for real-time face detection  
  - Pretrained lightweight model from [YOLOv8-face GitHub](https://github.com/derronqi/yolov8-face)  
  - Achieved 98.7% detection accuracy in testing

- **FaceNet** → Used for generating face embeddings and recognition  
  - Pretrained model based on TensorFlow/Keras  
  - Achieved 95.2% recognition accuracy in this project

