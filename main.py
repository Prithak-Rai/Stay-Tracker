import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import time

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

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    current_time = time.time()
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Initialize or update the detection time for known faces
        if name not in face_detection_times:
            face_detection_times[name] = current_time
        
        # Calculate elapsed time
        elapsed_time = current_time - face_detection_times[name]
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format the elapsed time as H:M:S
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        if name == "Unknown":
            b, g, r = 0, 0, 255
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
