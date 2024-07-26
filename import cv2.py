import cv2
import numpy as np
import datetime
import os
from collections import OrderedDict
from scipy.spatial import distance as dist

# Centroid Tracker class
class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bounding_boxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bounding_boxes[self.nextObjectID] = bbox  # Register bounding box
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bounding_boxes[objectID]  # Deregister bounding box
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bounding_boxes[objectID] = rects[col]  # Update bounding box
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], rects[col])

        return self.objects

# Initialize centroid tracker and start times dictionary
ct = CentroidTracker()
start_times = {}
snapshot_counts = {}
max_snapshots = 5
snapshot_folder = "snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

# Load pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('files/deploy.prototxt', 'files/mobilenet_iter_73000.caffemodel')

# Constants for the frame adjustment
FRAME_WIDTH, FRAME_HEIGHT = 600, 600
FOLLOW_PERSON_ID = None  # ID of the person to follow

# Video capture setup
cap = cv2.VideoCapture(0)  # Adjust as needed for your video source
# cap = cv2.VideoCapture('Vid/c.mp4')  # Uncomment and specify file path for video file
# cap = cv2.VideoCapture("http://192.168.0.103:8080/video")  # Uncomment and specify URL for IP camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    height, width = frame.shape[:2]

    # Prepare the input blob for the neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []

    # Process each detection from the neural network
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Adjust the index based on your object class
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                rects.append((startX, startY, endX, endY))

    # Update the centroid tracker with the detected bounding boxes
    objects = ct.update(rects)

    # Process each tracked object
    for (objectID, centroid) in objects.items():
        if FOLLOW_PERSON_ID is None:
            FOLLOW_PERSON_ID = objectID

        # Retrieve the bounding box coordinates
        (startX, startY, endX, endY) = ct.bounding_boxes[objectID]

        # Draw a rectangle around the detected person
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Initialize start time and snapshot count for the person if not already done
        if objectID not in start_times:
            start_times[objectID] = datetime.datetime.now()
            snapshot_counts[objectID] = 0

            # Create a folder for the person's snapshots
            person_folder = os.path.join(snapshot_folder, f"person_{objectID}")
            os.makedirs(person_folder, exist_ok=True)

            # Take a snapshot and save it
            snapshot_filename = os.path.join(person_folder, f"snapshot_{snapshot_counts[objectID]}.jpg")
            cv2.imwrite(snapshot_filename, frame)

            # Initialize duration text file with zero duration
            with open(os.path.join(person_folder, 'duration.txt'), 'w') as f:
                f.write('0d 0h 0m 0s')

        # Read existing duration from the file
        with open(os.path.join(person_folder, 'duration.txt'), 'r') as f:
            existing_duration = f.read().strip()
            days = int(existing_duration.split()[0][:-1])
            hours = int(existing_duration.split()[1][:-1])
            minutes = int(existing_duration.split()[2][:-1])
            seconds = int(existing_duration.split()[3][:-1])
            prev_duration = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

        # Calculate current duration
        current_duration = datetime.datetime.now() - start_times[objectID]
        total_duration = prev_duration + current_duration

        total_seconds = total_duration.total_seconds()
        days = total_seconds // (24 * 3600)
        hours = (total_seconds % (24 * 3600)) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        # Format duration text
        duration_text = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

        # Update the duration text file with the new duration
        with open(os.path.join(person_folder, 'duration.txt'), 'w') as f:
            f.write(duration_text)

        # Display the duration text above the detected person
        cv2.putText(frame, duration_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract a region of interest (ROI) around the person to follow if it's the designated person
        if objectID == FOLLOW_PERSON_ID:
            roi_startX = max(0, centroid[0] - FRAME_WIDTH // 2)
            roi_startY = max(0, centroid[1] - FRAME_HEIGHT // 2)
            roi_endX = min(width, roi_startX + FRAME_WIDTH)
            roi_endY = min(height, roi_startY + FRAME_HEIGHT)
            roi_startX = max(0, roi_endX - FRAME_WIDTH)
            roi_startY = max(0, roi_endY - FRAME_HEIGHT)
            frame = frame[roi_startY:roi_endY, roi_startX:roi_endX]


    # Display the frame with annotations
    cv2.imshow('Person Detection', frame)

    # Check for quit key ('q') to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()