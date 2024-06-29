import os
import cv2
import numpy as np
import datetime
from collections import OrderedDict
from scipy.spatial import distance as dist

# Centroid Tracker class
class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bounding_boxes = OrderedDict()  # Store bounding boxes
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.start_times = OrderedDict()  # Store start times for each person
        self.snapshot_count = OrderedDict()  # Store snapshot count for each person

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bounding_boxes[self.nextObjectID] = bbox  # Register bounding box
        self.disappeared[self.nextObjectID] = 0
        self.start_times[self.nextObjectID] = datetime.datetime.now()
        self.snapshot_count[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bounding_boxes[objectID]  # Deregister bounding box
        del self.disappeared[objectID]
        del self.start_times[objectID]
        del self.snapshot_count[objectID]

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

# Function to create folder if it does not exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Initialize centroid tracker
ct = CentroidTracker()
snapshots_dir = 'snapshots/'  # Path to snapshots directory

# Ensure snapshots directory exists
create_folder_if_not_exists(snapshots_dir)

# Load pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('files/deploy.prototxt', 'files/mobilenet_iter_73000.caffemodel')

# cap = cv2.VideoCapture('Vid/c.mp4')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.0.103:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (600, 600))
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # Retrieve the bounding box from the tracker
        (startX, startY, endX, endY) = ct.bounding_boxes[objectID]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        duration = datetime.datetime.now() - ct.start_times[objectID]
        days, seconds = duration.days, duration.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        total_duration_text = f"Total time: {days}d {hours}h {minutes}m {seconds}s"

        cv2.putText(frame, total_duration_text, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save total tracked time to text file
        person_folder = os.path.join(snapshots_dir, f"person_{objectID}")
        create_folder_if_not_exists(person_folder)
        time_filename = os.path.join(person_folder, "tracked_time.txt")
        with open(time_filename, 'w') as f:
            f.write(total_duration_text)

        # Capture and save snapshots 5 times
        if ct.snapshot_count[objectID] < 5:
            snapshot_filename = os.path.join(person_folder, f"snapshot_{ct.snapshot_count[objectID]}.jpg")
            cv2.imwrite(snapshot_filename, frame)
            ct.snapshot_count[objectID] += 1

    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
