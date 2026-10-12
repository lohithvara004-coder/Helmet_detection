

# STEP 1 - Install libraries
!pip install ultralytics roboflow deepface mtcnn cvzone

# STEP 2 - Imports
from ultralytics import YOLO
from mtcnn import MTCNN
from google.colab.patches import cv2_imshow
import cv2
import sqlite3
from datetime import datetime
import os
import glob

# STEP 3 - Download Dataset
from roboflow import Roboflow
rf = Roboflow(api_key="BxjKm43BarQ4J3QbzXZZ")
project = rf.workspace("pulsegen10").project("with-or-without-helmet-f7du9")
version = project.version(2)
dataset = version.download("yolov8")

# STEP 4 - Train Model
model = YOLO('yolov8n.pt')
model.train(
    data='/content/with-or-without-helmet-2/data.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    name='helmet_detection'
)
print("Training complete!")

# STEP 5 - Database Setup
os.makedirs('/content/violation_faces', exist_ok=True)
conn = sqlite3.connect('violations.db')
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS violations")
cursor.execute('''CREATE TABLE violations 
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                violation_type TEXT,
                camera_id TEXT,
                person TEXT,
                face_image TEXT)''')
conn.commit()
print("Database created!")

# STEP 6 - Load Model & MTCNN
model = YOLO('/content/runs/detect/helmet_detection2/weights/best.pt')
detector = MTCNN()

# STEP 7 - Detection on Images
results = model.predict(
    source='/content/with-or-without-helmet-2/test/images',
    save=True,
    conf=0.25
)
print("Detection complete!")

# STEP 8 - MTCNN Face Detection & Violation Logging
test_images = glob.glob('/content/with-or-without-helmet-2/test/images/*.jpg')[:10]
violation_count = 0

for img_path in test_images:
    img = cv2.imread(img_path)
    results = model.predict(img, conf=0.25, verbose=False)
    
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            if label == 'without_helmet':
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_img)
                person = "Unknown Violator"
                face_path = "No face detected"
                if faces:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    face_crop = img[y:y+h, x:x+w]
                    face_path = f'/content/violation_faces/violation_{violation_count}.jpg'
                    cv2.imwrite(face_path, face_crop)
                cursor.execute(
                    "INSERT INTO violations VALUES (NULL, ?, ?, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     label, 'CAM_01', person, face_path)
                )
                conn.commit()
                violation_count += 1

print(f"Total violations logged: {violation_count}")

# STEP 9 - Show Violations from Database
cursor.execute("SELECT * FROM violations")
rows = cursor.fetchall()
for row in rows:
    print(row)

# STEP 10 - Display Detected Images
images = glob.glob('/content/runs/detect/predict/*.jpg')[:5]
for img_path in images:
    img = cv2.imread(img_path)
    cv2_imshow(img)
    print("---")

# STEP 11 - Video Detection
cap = cv2.VideoCapture('/content/bike_1.mp4')
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 30 != 0:
        continue
    results = model.predict(frame, conf=0.10, verbose=False)
    annotated = results[0].plot()
    cv2_imshow(annotated)
    if frame_count > 150:
        break
cap.release()
print("Video Detection Done!")

# STEP 12 - Show Violation Faces
for i in range(violation_count):
    img = cv2.imread(f'/content/violation_faces/violation_{i}.jpg')
    if img is not None:
        print(f"Violation {i+1}:")
        cv2_imshow(img)
        print("---")