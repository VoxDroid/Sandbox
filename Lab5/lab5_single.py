from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model('test/images/0_1-157-_jpg.rf.147f96e1a074e34c8c5dae83b00e8ae9.jpg')  # Replace with your test image
results[0].show()  # Displays the image with detections