import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import time
import os

sfr = SimpleFacerec()
sfr.load_encoding_images("Images/")

cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera with index {i}")
        break

face_detection_times = {}
unknown_person_count = 0  

def save_snapshot(image, face_loc, count):
    y1, x2, y2, x1 = face_loc
    face_image = image[y1:y2, x1:x2]
    snapshot_folder = "Images"
    snapshot_path = os.path.join(snapshot_folder, f"unknown_person_{count}.jpg")
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

    face_locations, face_names = sfr.detect_known_faces(frame)
    
    current_time = time.time()
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

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
                snapshot_path = save_snapshot(frame, face_loc, unknown_person_count)
                unknown_image = face_recognition.load_image_file(snapshot_path)
                unknown_encodings = face_recognition.face_encodings(unknown_image)

                if unknown_encodings:
                    unknown_encoding = unknown_encodings[0]
                    matched_name = check_if_known(unknown_encoding, sfr.known_face_encodings, sfr.known_face_names)

                    if matched_name != "Unknown":
                        name = matched_name
                        os.remove(snapshot_path)  
                    else:
                        unknown_person_count += 1
                else:
                    print("No face found in the snapshot.")
                    os.remove(snapshot_path)  

        else:
            b, g, r = 0, 255, 0
        
        font_scale = 1.5
        cv2.putText(frame, f"{name} - {time_str}", (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (b, g, r), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()