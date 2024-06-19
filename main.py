import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('files/deploy.prototxt', 'files/mobilenet_iter_73000.caffemodel')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.0.103:8080/video")

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

    # Display the output frame
    cv2.imshow('Person Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
