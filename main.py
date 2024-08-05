import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import time
import os

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("Images/")

# Load Camera
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

# Dictionary to keep track of face detection start times
face_detection_times = {}
unknown_person_count = 0  # Counter for unknown persons

def save_snapshot(image, face_loc, elapsed_time):
    y1, x2, y2, x1 = face_loc
    face_image = image[y1:y2, x1:x2]
    snapshot_folder = "Images"
    
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{hours:02}h_{minutes:02}m_{seconds:02}s"
    
    snapshot_path = os.path.join(snapshot_folder, f"unknown_{time_str}.jpg")
    cv2.imwrite(snapshot_path, face_image)
    return snapshot_path

def check_if_known(face_encoding, known_encodings, known_names):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    if any(matches):
        match_index = matches.index(True)
        return known_names[match_index]
    return "Unknown"

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    current_time = time.time()
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        identifier = (x1, y1, x2, y2, name)

        if identifier not in face_detection_times:
            face_detection_times[identifier] = current_time
        
        elapsed_time = current_time - face_detection_times[identifier]
        
        if name == "Unknown":
            b, g, r = 0, 0, 255

            # Save snapshot and check if the person already exists
            if elapsed_time > 2:  # Wait for a couple of seconds to ensure a clear face
                snapshot_path = save_snapshot(frame, face_loc, elapsed_time)
                unknown_image = face_recognition.load_image_file(snapshot_path)
                unknown_encodings = face_recognition.face_encodings(unknown_image)

                if unknown_encodings:
                    unknown_encoding = unknown_encodings[0]
                    matched_name = check_if_known(unknown_encoding, sfr.known_face_encodings, sfr.known_face_names)

                    if matched_name != "Unknown":
                        name = matched_name
                        os.remove(snapshot_path)  # Remove snapshot if the person is known
                    else:
                        unknown_person_count += 1
                else:
                    print("No face found in the snapshot.")
                    os.remove(snapshot_path)  # Remove snapshot if no face is found

        else:
            b, g, r = 0, 255, 0
        
        font_scale = 1.5
        cv2.putText(frame, f"{name} - {elapsed_time:.0f}s", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
