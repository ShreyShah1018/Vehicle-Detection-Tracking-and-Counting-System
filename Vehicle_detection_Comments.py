import cv2  # Importing OpenCV library for image and video processing
from sort import *  # Importing SORT algorithm for object tracking
import math  # Importing math module for mathematical operations
import numpy as np  # Importing NumPy library for array operations
from ultralytics import YOLO  # Importing YOLO object detection model from Ultralytics
import cvzone  # Importing cvzone library for drawing utilities

# Opening video file for processing
cap = cv2.VideoCapture('cars2.mp4')

# Loading YOLOv8 nano model for object detection
model = YOLO('yolov8n.pt')

# Reading class names from file
classnames = []
with open('new_classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Initializing SORT tracker
tracker = Sort(max_age=20)

# Defining a line for counting vehicles
line = [320, 350, 620, 350]  #To be changed if the video is altered
counter = []

# Main loop for processing video frames
while 1:
    # Reading a frame from the video
    ret, frame = cap.read()

    # If frame is not retrieved, restart the video
    if not ret:
        cap = cv2.VideoCapture('cars2.mp4')
        continue

    # Array to store detections
    detections = np.empty((0, 5))

    # Running YOLO model for object detection on the frame
    result = model(frame, stream=1)

    # Processing YOLO output
    for info in result:
        boxes = info.boxes
        for box in boxes:
            # Extracting bounding box coordinates, confidence, and class index
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            # If object is a car, bus, or truck with confidence > 60%, add to detections
            if objectdetect == 'car' or objectdetect == 'bus' or objectdetect == 'truck' and conf > 60:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

    # Update tracker with detections
    track_result = tracker.update(detections)

    # Draw line for counting vehicles
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 7)

    # Processing tracker output
    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        # Calculating center of the bounding box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Drawing bounding box and ID
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cvzone.putTextRect(frame, f'{objectdetect}{id}  ={conf}%', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        # Counting vehicles crossing the line
        if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 15)
            if counter.count(id) == 0:
                counter.append(id)

    # Displaying total count of vehicles
    cvzone.putTextRect(frame, f'Total Vehicles ={len(counter)}', [290, 34], thickness=4, scale=2.3, border=2)

    # Displaying frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

import matplotlib.pyplot as plt

# Example data (replace with your actual metric values)
epochs = [1, 2, 3, 4, 5]
box_loss_values = [0.1, 0.08, 0.06, 0.05, 0.04]

# Create a line chart
plt.figure(figsize=(8, 6))
plt.plot(epochs, box_loss_values, marker='o', label='Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Box Loss')
plt.grid(True)
plt.legend()
plt.show()