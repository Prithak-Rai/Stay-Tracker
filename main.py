import cv2
import face_recognition
import sqlite3
import time
import numpy as np

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

def save_snapshot(image, face_loc):
    y1, x2, y2, x1 = face_loc
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
    cursor.execute("UPDATE photos SET last_seen = ? WHERE person_id = ?", (last_seen, person_id))
    conn.commit()

def update_timestamp(cursor, conn, person_id, timestamp):
    cursor.execute("UPDATE photos SET timestamp = ? WHERE person_id = ?", (timestamp, person_id))
    conn.commit()

def retrieve_last_seen(cursor, person_id):
    cursor.execute("SELECT last_seen FROM photos WHERE person_id = ?", (person_id,))
    result = cursor.fetchone()
    return result[0] if result else None

def main():
    conn = connect_db()
    cursor = conn.cursor()
    known_face_encodings, known_face_names = load_encodings_from_db(cursor)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found or cannot be opened.")
        conn.close()
        return

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

            current_time = time.time()
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            current_detected_person_ids = []

            for face_encoding, face_loc in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                else:
                    unknown_person_count += 1

                person_id = get_or_create_person(cursor, conn, name)
                if name == "Unknown":
                    image_blob = save_snapshot(frame, face_loc)
                    store_photo_data(cursor, conn, person_id, timestamp_str, "0:00:00", image_blob)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                else:
                    update_last_seen(cursor, conn, person_id, timestamp_str)

                current_detected_person_ids.append(person_id)

                if person_id not in face_detection_times:
                    face_detection_times[person_id] = current_time  
                    last_seen_times[person_id] = current_time 

                last_seen_times[person_id] = current_time

                elapsed_time = current_time - face_detection_times[person_id]
                hours, remainder = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

                update_timestamp(cursor, conn, person_id, time_str)

                b, g, r = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 480
                y1, x2, y2, x1 = [int(coord * scale) for coord, scale in zip(face_loc, [scale_y, scale_x, scale_y, scale_x])]

                cv2.putText(frame, f"{name} - {time_str}", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (b, g, r), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)

            for person_id in list(last_seen_times.keys()):
                if person_id not in current_detected_person_ids:
                    if current_time - last_seen_times[person_id] > timeout_duration:
                        del face_detection_times[person_id] 
                        del last_seen_times[person_id]  

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()

if __name__ == "__main__":
    main()
