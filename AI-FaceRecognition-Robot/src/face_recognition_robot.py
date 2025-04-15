import cv2
import torch
import time
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load camera
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Paths
data_dir = "Data"
saved_images_dir = "Unknown"

if not os.path.exists(saved_images_dir):
    os.makedirs(saved_images_dir)

# Load known face embeddings and names
known_face_embeddings = []
known_face_names = []

for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    if os.path.isdir(person_folder_path):
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Load image and compute embeddings
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces, _ = mtcnn.detect(rgb_image)

                if faces is not None:
                    for box in faces:
                        x1, y1, x2, y2 = map(int, box)
                        face_image = rgb_image[y1:y2, x1:x2]
                        face_tensor = preprocess(face_image).unsqueeze(0).to(device)
                        embedding = resnet(face_tensor).detach().cpu().numpy()
                        known_face_embeddings.append(embedding)
                        known_face_names.append(person_folder)

# Main loop
detect_interval = 5
frame_count = 0
fps_list = []

# Initialize variables to handle detection
faces = None
probs = None

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame_count += 1

    # Face detection every 'detect_interval' frames
    if frame_count % detect_interval == 0:
        small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        faces, probs = mtcnn.detect(small_frame_rgb)

        if faces is not None:
            faces = faces * 2  # Scale back up
        else:
            probs = None

    # Draw bounding boxes and process detected faces
    if faces is not None and probs is not None:
        for i, box in enumerate(faces):
            if probs[i] < 0.9: 
                continue

            x1, y1, x2, y2 = map(int, box)
            face_image = frame[y1:y2, x1:x2]
            face_tensor = preprocess(face_image).unsqueeze(0).to(device)
            face_embedding = resnet(face_tensor).detach().cpu().numpy()

            # Compare embeddings
            name = "Unknown"
            color = (0, 0, 255)
            min_dist = float("inf")

            for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
                dist = np.linalg.norm(face_embedding - known_embedding)
                if dist < 0.6 and dist < min_dist:  # Threshold
                    min_dist = dist
                    name = known_name
                    color = (0, 255, 0)

            # Save unknown face
            if name == "Unknown":
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(saved_images_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved unknown face: {filepath}")

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time) if end_time > start_time else 0
    fps_list.append(fps)
    if len(fps_list) > 30:  # Smooth FPS over last 30 frames
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    # Display FPS
    fps_text = f"FPS: {avg_fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
