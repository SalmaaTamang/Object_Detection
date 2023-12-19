from ultralytics import YOLO
import cv2
img =cv2.imread('images/car.jpg')
new_width=512
new_height=512
resized_img=cv2.resize(img, (new_width, new_height))
# cv2.imshow('original img',img)
cv2.imshow('resized img',resized_img)
model=YOLO('yolov8n.pt')
result=model(resized_img, show=True)
cv2.waitKey(0)