import cv2
import face_recognition
import sqlite3
import time
import numpy as np

def connect_db():
    return sqlite3.connect('face_data.db')

def load_encodings_from_db(cursor):
    cursor.execute("SELECT name, image FROM face_data")
    encodings = []
    names = []
    for row in cursor.fetchall():
        name = row[0]
        image_blob = row[1]
        image_array = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
        face_encodings = face_recognition.face_encodings(image_array)
        if face_encodings:
            encodings.append(face_encodings[0])
            names.append(name)
    return encodings, names

def save_snapshot(image, face_loc, original_shape):
    y1, x2, y2, x1 = face_loc
    scale_x = original_shape[1] / 640
    scale_y = original_shape[0] / 480
    x1, x2, y1, y2 = [int(coord * scale) for coord, scale in zip([x1, x2, y1, y2], [scale_x, scale_x, scale_y, scale_y])]
    
    x1 = max(0, x1)
    x2 = min(original_shape[1], x2)
    y1 = max(0, y1)
    y2 = min(original_shape[0], y2)
    
    face_image = image[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', face_image)
    return buffer.tobytes()

def store_face_data(cursor, name, timestamp, image_blob):
    cursor.execute("INSERT INTO face_data (name, timestamp, image) VALUES (?, ?, ?)", 
                   (name, timestamp, image_blob))
    conn.commit()

def update_timestamp(cursor, name, timestamp):
    cursor.execute("UPDATE face_data SET timestamp = ? WHERE name = ?", (timestamp, name))
    conn.commit()

conn = connect_db()
cursor = conn.cursor()
known_face_encodings, known_face_names = load_encodings_from_db(cursor)

cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

if not cap.isOpened():
    print("Camera not found or cannot be opened.")
    conn.close()
    exit()

face_detection_times = {}
unknown_person_count = 0  

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        original_shape = frame.shape
        frame_small = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_time = time.time()
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                update_timestamp(cursor, name, timestamp_str)
            else:
                unknown_person_count += 1
                image_blob = save_snapshot(frame, face_loc, original_shape)  # Pass original_shape here
                store_face_data(cursor, name, timestamp_str, image_blob)
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

            scale_x = original_shape[1] / 640
            scale_y = original_shape[0] / 480
            y1, x2, y2, x1 = [int(coord * scale) for coord, scale in zip(face_loc, [scale_y, scale_x, scale_y, scale_x])]
            
            identifier = (x1, y1, x2, y2, name)

            if identifier not in face_detection_times:
                face_detection_times[identifier] = current_time

            elapsed_time = current_time - face_detection_times[identifier]
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

            b, g, r = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

            font_scale = 1.5
            cv2.putText(frame, f"{name} - {time_str}", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
