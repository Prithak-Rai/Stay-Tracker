import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        valid_extensions = ('.png', '.jpg', '.jpeg')  # Add any other image file extensions if needed
        images_path = os.path.join(images_path)
        image_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and f.lower().endswith(valid_extensions)]

        for image_filename in image_filenames:
            try:
                image_path = os.path.join(images_path, image_filename)
                
                # Load image using OpenCV to ensure it's in correct format
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error reading file {image_filename}: Unable to read image.")
                    continue
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get face encodings
                encoding = face_recognition.face_encodings(rgb_image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    self.known_face_names.append(os.path.splitext(image_filename)[0])
            except Exception as e:
                print(f"Error processing file {image_filename}: {e}")

        print(f"{len(self.known_face_encodings)} encoding images found.")
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names
