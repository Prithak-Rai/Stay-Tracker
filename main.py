import cv2
import face_recognition
import sqlite3
import time
import numpy as np
from datetime import datetime, timedelta

def connect_db():
    return sqlite3.connect('face_data.db')

def load_encodings_from_db(cursor):
    cursor.execute("SELECT p.id, p.name, ph.image FROM person p INNER JOIN photos ph ON p.id = ph.person_id")
    encodings = []
    names = []
    for row in cursor.fetchall():
        person_id, name, image_blob = row
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

def get_or_create_person(cursor, conn, name):
    cursor.execute("SELECT id FROM person WHERE name = ?", (name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute("INSERT INTO person (name) VALUES (?)", (name,))
    conn.commit()
    return cursor.lastrowid  

def store_photo_data(cursor, conn, person_id, last_seen, timestamp, image_blob):
    cursor.execute("INSERT INTO photos (person_id, last_seen, timestamp, image) VALUES (?, ?, ?, ?)", 
                   (person_id, last_seen, timestamp, image_blob))
    conn.commit()

def update_last_seen(cursor, conn, person_id, last_seen):
    cursor.execute("UPDATE person SET last_seen = ? WHERE id = ?", (last_seen, person_id))
    cursor.execute("UPDATE photos SET last_seen = ? WHERE id = ?", (last_seen, person_id))
    conn.commit()

def update_timestamp(cursor, conn, person_id, timestamp):

    try:
        t1 = datetime.strptime(timestamp, '%H:%M:%S') - datetime(1900, 1, 1)
    except ValueError as e:
        print(f"Timestamp format error: {e}")
        return

    print(f"New timestamp (t1): {t1}")

    cursor.execute("SELECT timestamp FROM photos WHERE person_id = ? ORDER BY id DESC LIMIT 1", (person_id,))
    last_timestamp = cursor.fetchone()
    
    if last_timestamp and last_timestamp[0]:
        try:
            x = datetime.strptime(last_timestamp[0], '%H:%M:%S') - datetime(1900, 1, 1)
        except ValueError as e:
            print(f"Stored timestamp format error: {e}")
            x = timedelta(0)
    else:
        x = timedelta(0)

    print(f"Last known timestamp (x): {x}")

    if t1 > x:

        cursor.execute("SELECT total_time_spent FROM person WHERE id = ?", (person_id,))
        t2 = cursor.fetchone()
        print(f"Current total_time_spent from DB: {t2}")

        if t2 and t2[0]:
            try:
                total = datetime.strptime(t2[0], '%H:%M:%S') - datetime(1900, 1, 1)
            except ValueError as e:
                print(f"Stored total_time_spent format error: {e}")
                total = timedelta(0)
        else:
            total = timedelta(0)

        print(f"Total time spent (total): {total}")

        total_time = total + timedelta(seconds=1)  
        print(f"Updated total time (total_time): {total_time}")

        total_seconds = int(total_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        str_total = f'{hours:02}:{minutes:02}:{seconds:02}'
        print(f"Formatted total_time_spent: {str_total}")

        try:
            cursor.execute("UPDATE photos SET timestamp = ? WHERE person_id = ?", (timestamp, person_id))
            cursor.execute("UPDATE person SET total_time_spent = ? WHERE id = ?", (str_total, person_id))
            conn.commit()
        except Exception as e:
            print(f"Database update error: {e}")
    else:
        print("No change in timestamp; total_time_spent remains unchanged.")


def retrieve_last_seen(cursor, person_id):
    cursor.execute("SELECT last_seen FROM person WHERE id = ?", (person_id,))
    result = cursor.fetchone()
    return result[0] if result else '0'

def calculate_duration(last_seen, current_time):
    if last_seen == '0':
        return 0
    last_seen_time = datetime.strptime(last_seen, '%Y-%m-%d %H:%M:%S')
    current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    duration = current_time - last_seen_time
    return int(duration.total_seconds())

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
last_seen_times = {}
unknown_person_count = 0  
timeout_duration = 2  

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

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        current_detected_person_ids = []

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                person_id = get_or_create_person(cursor, conn, name)
                last_seen = retrieve_last_seen(cursor, person_id)
                update_last_seen(cursor, conn, person_id, current_time)
            else:
                unknown_person_count += 1
                person_id = get_or_create_person(cursor, conn, name)
                image_blob = save_snapshot(frame, face_loc, original_shape)
                store_photo_data(cursor, conn, person_id, current_time, "0:00:00", image_blob)
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

            current_detected_person_ids.append(person_id)

            if person_id not in face_detection_times:
                face_detection_times[person_id] = time.time()  
                last_seen_times[person_id] = time.time() 

            last_seen_times[person_id] = time.time()

            elapsed_time = last_seen_times[person_id] - face_detection_times[person_id]
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

            update_timestamp(cursor, conn, person_id, time_str)

            duration = calculate_duration(retrieve_last_seen(cursor, person_id), current_time)

            b, g, r = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

            scale_x = original_shape[1] / 640
            scale_y = original_shape[0] / 480
            y1, x2, y2, x1 = [int(coord * scale) for coord, scale in zip(face_loc, [scale_y, scale_x, scale_y, scale_x])]

            font_scale = 1.5
            cv2.putText(frame, f"{name} - {time_str}", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)

        for person_id in list(last_seen_times.keys()):
            if person_id not in current_detected_person_ids:
                if time.time() - last_seen_times[person_id] > timeout_duration:
                    del face_detection_times[person_id] 
                    del last_seen_times[person_id]  

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close