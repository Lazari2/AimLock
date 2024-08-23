import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow('Detecção em Tempo Real - YOLO', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()