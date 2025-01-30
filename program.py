from ultralytics import YOLO
import cv2
import math
import os
import time

# Webcam configuration (low resolution for optimization)
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 10  
FRAME_INTERVAL = 1 / TARGET_FPS

# Folder to save detected videos
SAVE_FOLDER = "hasil deteksi"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Load YOLO model
MODEL_PATH = "ObjekV8.pt"
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

# Class names for the objects (sesuaikan dengan kelas yang ada di model)
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Warna untuk tiap kelas
colors = {
    'person': (0, 255, 0),          # Hijau
    'bicycle': (0, 0, 255),         # Merah
    'car': (255, 0, 0),             # Biru
    'motorcycle': (0, 255, 255),    # Kuning
    'airplane': (255, 0, 255),      # Ungu
    'bus': (0, 255, 255),           # Kuning
    'train': (0, 255, 255),         # Kuning
    'truck': (255, 165, 0),         # Oranye
    'boat': (255, 255, 255),        # Putih
    'traffic light': (255, 255, 0), # Kuning
    'fire hydrant': (255, 0, 255),  # Ungu
    'stop sign': (0, 0, 255),       # Merah
    'parking meter': (0, 0, 255),   # Merah
    'bench': (255, 255, 255),       # Putih
    'bird': (255, 0, 0),            # Biru
    'cat': (0, 255, 0),             # Hijau
    'dog': (0, 0, 255),             # Merah
    'horse': (255, 255, 0),         # Kuning
    'sheep': (0, 255, 255),         # Kuning
    'cow': (255, 165, 0),           # Oranye
    'elephant': (255, 255, 255),    # Putih
    'bear': (255, 0, 0),            # Biru
    'zebra': (0, 255, 255),         # Kuning
    'giraffe': (255, 0, 255),       # Ungu
    'backpack': (255, 165, 0),      # Oranye
    'umbrella': (0, 0, 255),        # Merah
    'handbag': (255, 0, 255),       # Ungu
    'tie': (255, 255, 0),           # Kuning
    'suitcase': (0, 255, 0),        # Hijau
    'frisbee': (255, 255, 255),     # Putih
    'skis': (255, 0, 255),          # Ungu
    'snowboard': (0, 0, 255),       # Merah
    'sports ball': (255, 255, 0),   # Kuning
    'kite': (255, 0, 0),            # Biru
    'baseball bat': (0, 255, 0),    # Hijau
    'baseball glove': (255, 165, 0),# Oranye
    'skateboard': (255, 255, 0),    # Kuning
    'surfboard': (0, 255, 255),     # Kuning
    'tennis racket': (255, 0, 0),   # Biru
    'bottle': (0, 255, 0),          # Hijau
    'wine glass': (255, 0, 255),    # Ungu
    'cup': (0, 255, 255),           # Kuning
    'fork': (255, 0, 0),            # Biru
    'knife': (0, 0, 255),           # Merah
    'spoon': (255, 165, 0),         # Oranye
    'bowl': (255, 255, 255),        # Putih
    'banana': (255, 255, 0),        # Kuning
    'apple': (0, 255, 0),           # Hijau
    'sandwich': (255, 0, 255),      # Ungu
    'orange': (255, 165, 0),        # Oranye
    'broccoli': (0, 255, 255),      # Kuning
    'carrot': (255, 0, 0),          # Biru
    'hot dog': (0, 255, 0),         # Hijau
    'pizza': (255, 0, 255),         # Ungu
    'donut': (255, 165, 0),         # Oranye
    'cake': (0, 0, 255),            # Merah
    'chair': (255, 255, 255),       # Putih
    'couch': (0, 255, 255),         # Kuning
    'potted plant': (0, 255, 0),    # Hijau
    'bed': (255, 0, 0),             # Biru
    'dining table': (255, 0, 255),  # Ungu
    'toilet': (0, 255, 0),          # Hijau
    'tv': (255, 165, 0),            # Oranye
    'laptop': (255, 255, 0),        # Kuning
    'mouse': (0, 0, 255),           # Merah
    'remote': (0, 255, 255),        # Kuning
    'keyboard': (255, 0, 0),        # Biru
    'cell phone': (255, 255, 255),  # Putih
    'microwave': (255, 0, 255),     # Ungu
    'oven': (0, 255, 0),            # Hijau
    'toaster': (0, 255, 255),       # Kuning
    'sink': (255, 165, 0),          # Oranye
    'refrigerator': (0, 0, 255),    # Merah
    'book': (0, 255, 0),            # Hijau
    'clock': (255, 0, 255),         # Ungu
    'vase': (255, 255, 0),          # Kuning
    'scissors': (0, 0, 255),        # Merah
    'teddy bear': (255, 255, 255),  # Putih
    'hair drier': (255, 0, 0),      # Biru
    'toothbrush': (255, 165, 0),    # Oranye
}

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
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Handle the case when cls is out of bounds
                if cls < len(classNames):
                    class_name = classNames[cls]
                    color = colors.get(class_name, (255, 255, 255))  # Jika tidak ada warna, gunakan putih
                    label = f"{class_name} {confidence}"

                    # Draw bounding box with thicker line
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # Background for better text visibility
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)

                    # Put text on top of rectangle
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the result
        cv2.imshow('Webcam', img)

        # Write the frame to the output video
        out.write(img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
