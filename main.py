import cv2
import numpy as np
import datetime

# Load pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('files/deploy.prototxt', 'files/mobilenet_iter_73000.caffemodel')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.0.103:8080/video")

# Initialize variables to track detection duration
person_detected = False
start_time = None
end_time = None

while cap.isOpened():
    # Read the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to improve processing speed
    frame = cv2.resize(frame, (600, 600))
    height, width = frame.shape[:2]

    # Preprocess the frame for the neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_in_frame = False

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.5:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # If the detection is a person (class label 15 in COCO dataset for MobileNet SSD)
            if idx == 15:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                detected_in_frame = True

                # Display the duration above the rectangle if a person was detected previously
                if person_detected:
                    duration = datetime.datetime.now() - start_time
                    days, seconds = duration.days, duration.seconds
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    seconds = seconds % 60
                    duration_text = f"{days}d {hours}h {minutes}m {seconds}s"
                    cv2.putText(frame, duration_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update detection duration tracking
    if detected_in_frame:
        if not person_detected:
            start_time = datetime.datetime.now()
            person_detected = True
    else:
        if person_detected:
            person_detected = False

    # Display the output frame
    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()