# Author: Yash Raj Mani
# Notes:
#   Goes frame by frame
#   Struggles to resolve individuals as one person, often misses entire people
#   Least effective in testing
# Link to original repo: https://github.com/yashrajmani/OpenCV_Yolo3_Object_Detection-from-Video/tree/main

import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolo-coco-data/yolov3.weights", "yolo-coco-data/yolov3.cfg")
classes = []

# Load class names from coco file
with open("cfg/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load videos
video_path = "videos/video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables for frame extraction
frame_rate = 60  # Extract one frame per minute
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_rate == 0:
        # Preprocess the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Process YOLO output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Extract detection details
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw a box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("DISPLAYING: FRAME | Detections", frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()