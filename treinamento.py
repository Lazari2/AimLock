from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='C:/opencv/dataset.yaml', epochs=200, imgsz=640)