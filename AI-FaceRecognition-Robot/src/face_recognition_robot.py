import sqlite3
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import torchvision.transforms as transforms
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Connect to DB and load known embeddings
conn = sqlite3.connect("faces.db")
cursor = conn.cursor()
known_embeddings = []
known_names = []

cursor.execute("""
    SELECT person.id, person.name, faces.image
    FROM person
    JOIN faces ON person.id = faces.person_id
""")
rows = cursor.fetchall()

for person_id, person_name, image_blob in rows:
    try:
        image = Image.open(io.BytesIO(image_blob)).convert("RGB")
        face_tensor = preprocess(np.array(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()
        known_embeddings.append(embedding)
        known_names.append(person_name)
    except Exception as e:
        print(f"Error loading image for {person_name}: {e}")

print(f"✅ Loaded {len(known_names)} face images from DB.")

# Email setup
EMAIL_ADDRESS = "prithak.khamtu@gmail.com"
EMAIL_PASSWORD = "paykcwhdbymsukrk"
RECEIVER_EMAIL = "prithakhamtu@gmail.com" 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email_with_image(image_path, subject="Unknown Person Detected"):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText("An unknown person was detected by your security system.", 'plain'))
    with open(image_path, 'rb') as f:
        img_data = f.read()
    msg.attach(MIMEImage(img_data, name=os.path.basename(image_path)))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email sent with image: {image_path}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

# Optional: reduce resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

saved_unknown_dir = "Unknown"
if not os.path.exists(saved_unknown_dir):
    os.makedirs(saved_unknown_dir)

# Unknown face tracking
unknown_face_detected_time = None
unknown_face_detected = False
unknown_face_delay = 0.5  # Seconds

# FPS control
prev_time = 0
frame_rate = 15  # Target FPS

while True:
    current_time = time.time()
    if current_time - prev_time < 1.0 / frame_rate:
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        faces, probs = mtcnn.detect(rgb_small_frame)

    unknown_face_in_frame = False

    if faces is not None:
        for i, box in enumerate(faces):
            if probs[i] < 0.90:
                continue
            x1, y1, x2, y2 = [int(coord * 2) for coord in box]  # Scale back to original frame
            face_img = frame[y1:y2, x1:x2]

            try:
                face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    face_embedding = resnet(face_tensor).cpu().numpy()

                name = "Unknown"
                for emb, known_name in zip(known_embeddings, known_names):
                    similarity = cosine_similarity(face_embedding, emb)
                    distance = 1 - similarity
                    if distance < 0.5:
                        name = known_name
                        break

                # Handle unknown face
                if name == "Unknown":
                    unknown_face_in_frame = True
                    if not unknown_face_detected:
                        unknown_face_detected = True
                        unknown_face_detected_time = current_time
                        unknown_face_coords = (x1, y1, x2, y2)
                    elif current_time - unknown_face_detected_time >= unknown_face_delay:
                        flag_path = f"sent_{int(unknown_face_detected_time)}.flag"
                        if not os.path.exists(flag_path):
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"Unknown_{timestamp}.jpg"
                            filepath = os.path.join(saved_unknown_dir, filename)
                            cv2.imwrite(filepath, frame)
                            print(f"📸 Saved unknown face: {filename}")
                            send_email_with_image(filepath)
                            open(flag_path, 'w').close()
                else:
                    unknown_face_detected = False

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                print(f"Face processing error: {e}")

    if not unknown_face_in_frame:
        unknown_face_detected = False

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()