import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 nano model (lightweight and fast)
model = YOLO('runs/detect/train/weights/best.pt')  # Use trained model
# model = YOLO('yolov8n.pt')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Draw bounding boxes, labels, and confidences on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()