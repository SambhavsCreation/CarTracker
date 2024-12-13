import os

import cv2
import numpy as np

# Load pre-trained vehicle detector model using YOLO
YOLO_CONFIG = 'yolov4.cfg'  # Path to YOLO configuration file
YOLO_WEIGHTS = 'yolov4.weights'  # Path to YOLO pre-trained weights
YOLO_CLASSES = 'coco.names'  # Path to file with class names

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(YOLO_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def detect_and_track_vehicles(video_path):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was successfully opened
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        while True:
            # Read frames from the video
            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            # Get frame dimensions
            height, width, channels = frame.shape

            # Prepare the frame for YOLO model
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(output_layers)

            # Initialize lists for detected vehicles
            class_ids = []
            confidences = []
            boxes = []

            # Parse detections
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.01 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Maximum Suppression to filter boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)  # Green for bounding boxes
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the frame with detections
            cv2.imshow('Vehicle Detector', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close windows
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

# Path to the local video file
video_path = "videoplayback.mp4"

# Run the vehicle detection and tracking function
detect_and_track_vehicles(video_path)
