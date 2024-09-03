import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import time
import os
import sqlite3

# Initialize the face recognition and encoding module
sfr = SimpleFacerec()
sfr.load_encoding_images("Images/")

# Set up camera capture
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

if not cap.isOpened():
    print("Camera not found or cannot be opened.")
    exit()

# Initialize SQLite connection and cursor
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS face_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    timestamp TEXT,
                    image BLOB
                  )''')

# Function to save face snapshot as BLOB
def save_snapshot(image, face_loc):
    y1, x2, y2, x1 = face_loc
    face_image = image[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', face_image)
    return buffer.tobytes()

# Function to store face data in the database
def store_face_data(name, timestamp, image_blob):
    cursor.execute("INSERT INTO face_data (name, timestamp, image) VALUES (?, ?, ?)", 
                   (name, timestamp, image_blob))
    conn.commit()

face_detection_times = {}
unknown_person_count = 0  

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect known faces
    face_locations, face_names = sfr.detect_known_faces(rgb_frame)

    current_time = time.time()
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc
        identifier = (x1, y1, x2, y2, name)

        if identifier not in face_detection_times:
            face_detection_times[identifier] = current_time

        elapsed_time = current_time - face_detection_times[identifier]
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        if name == "Unknown":
            b, g, r = 0, 0, 255

            if elapsed_time > 2: 
                face_encodings = face_recognition.face_encodings(rgb_frame, [face_loc])

                if face_encodings:
                    unknown_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(sfr.known_face_encodings, unknown_encoding)
                    if any(matches):
                        match_index = matches.index(True)
                        name = sfr.known_face_names[match_index]
                    else:
                        unknown_person_count += 1
                        image_blob = save_snapshot(frame, face_loc)
                        store_face_data(name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)), image_blob)
        else:
            b, g, r = 0, 255, 0

        font_scale = 1.5
        cv2.putText(frame, f"{name} - {time_str}", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
conn.close()
