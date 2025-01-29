from ultralytics import YOLO
import cv2
import math
import os
import time

# Webcam configuration (low resolution for optimization)
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 60  # Set lower FPS to match hardware capability
FRAME_INTERVAL = 1 / TARGET_FPS

# Folder to save detected videos
SAVE_FOLDER = "hasil deteksi"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Load YOLO model
MODEL_PATH = "yolo-Weights/ObjekV8.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam.")

cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

last_frame_time = time.time()

# Class names for the objects (just example classes, you can add more)
classNames = ["knife", "bowl", "cup", "bottle"]

# Set up the VideoWriter to save the detected video
current_time_str = time.strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(SAVE_FOLDER, f"detected_video_{current_time_str}.mp4")

# Video writer with a specific codec (XVID) and FPS
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, TARGET_FPS, (CAM_WIDTH, CAM_HEIGHT))

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Exiting...")
            break

        # Limit frame processing based on time
        current_time = time.time()
        if current_time - last_frame_time < FRAME_INTERVAL:
            continue

        last_frame_time = current_time

        # Perform object detection
        results = model(img)

        # Process each detection result
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Ensure the coordinates are converted to list
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Handle the case when cls is out of bounds
                if cls < len(classNames):
                    label = f"{classNames[cls]} ({confidence})"
                else:
                    label = f"Unknown Class ({confidence})"  # Fallback for out-of-range class index

                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the result
        cv2.imshow('Webcam', img)

        # Write the frame to the output video
        out.write(img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
            print("Exiting...")
            break

finally:
    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()
