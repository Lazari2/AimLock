import cv2
from ultralytics import YOLO

model = YOLO('C:/opencv/runs/detect/train/weights/best.pt')

img = cv2.imread('C:/opencv/Img/train/images/7.png')


img = cv2.resize(img, (800, 640))  


results = model(img)

for result in results:
    for box in result.boxes:
        if box.conf > 0.7:
            print(f"Objeto detectado em: {box.xyxy}")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 

cv2.imshow('Detecção YOLO', img)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
