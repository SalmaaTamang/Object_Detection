from ultralytics import YOLO
import cv2
import cvzone 
import math

cap = cv2.VideoCapture(0)  # 0 for one webcam, 1 for multiple webcams
cap.set(3, 1280)  # width
cap.set(4, 720)  # height 630 by 480

model = YOLO("yoloweights/yolov8n.pt")


class_names = [
    'person', 'car', 'dog', 'cat', 'tree',
    'building', 'computer', 'book', 'chair', 'table',
    'phone', 'bicycle', 'pizza', 'coffee cup', 'pen',
    'airplane', 'flower', 'bird', 'traffic light', 'stop sign',
    'wallet', 'guitar', 'headphone', 'watch', 'shoe',
    'backpack', 'umbrella', 'sunglasses', 'hat', 'clock',
    'keyboard', 'mouse', 'banana', 'apple', 'orange',
    'bus', 'train', 'truck', 'motorcycle', 'boat','purse'
]

while True:
    success, img = cap.read()
    result = model(img, stream=True)

    # bounding box
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{class_names[cls]} {conf}', (max(0, x1), max(40, y1)))

    cv2.imshow("img", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
