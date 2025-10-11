if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=100, imgsz=640, device=0)