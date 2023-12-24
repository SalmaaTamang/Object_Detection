# import cv2
import numpy as np
# from tensorflow.keras.models import load_model

# model=load_model('mnist_model.h5')

# classes={0:'one',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}

# def img_classifier(img):

#   img=cv2.resize(img,(224,224))
#   img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#   img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#   img=img.reshape(1,224,224,3)
#   img=img/255.0

#   predication=model.predict(img)
#   class_index=np.argmax(predication)
#   class_label=classes[class_index]
#   confidence=predication[0][class_index]
#   return  class_label,confidence

# cap=cv2.VideoCapture(0)
# while True:
#   success,frame=cap.read(0)

#   if not success:
#     break
#   class_label,confidence=img_classifier(frame)  
#   cv2.putText(frame, f'Digit: {class_label} - Confidence: {confidence:.2f}', (10,30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#   cv2.imshow('Video', frame)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()        


# import cv2
# import cvzone 
# import math
# from tensorflow.keras.models import load_model

# model=load_model('mnist_model.h5')

# cap = cv2.VideoCapture(0)  # 0 for one webcam, 1 for multiple webcams
# cap.set(3, 1280)  # width
# cap.set(4, 720)  # height 630 by 480

# class_names = {0:'one',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}

# def img_classifier(img):

#   img=cv2.resize(img,(224,224))
#   img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#   img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#   img=img.reshape(1,224,224,3)
#   img=img/255.0
#   predication=model.predict(img)
#   class_index=np.argmax(predication)
#   class_label=class_names[class_index]
#   confidence=predication[0][class_index]
#   return  class_label,confidence,predication


# while True:
#     success, frame = cap.read()
#     class_label,cofidence,predication = img_classifier(frame)

#     # bounding box
#     for r in predication:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#             print(x1, y1, x2, y2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # # confidence
#             # conf = math.ceil((box.conf[0] * 100)) / 100

#             # # class name
#             # cls = int(box.cls[0])
#             cvzone.putTextRect(frame, f'{class_label} {confidence}', (max(0, x1), max(40, y1)))

#     cv2.imshow("img", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('mnist_model.h5')

cap = cv2.VideoCapture(0)  # 0 for one webcam, 1 for multiple webcams
cap.set(3, 1280)  # width
cap.set(4, 720)  # height 630 by 480

class_names = {0: 'one', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

def img_classifier(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.reshape(1, 224, 224, 3)
    img = img / 255.0
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = prediction[0][class_index]
    return class_label, confidence, prediction

# Create a CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Initialize bounding box
bbox = (50, 50, 200, 200)
tracking = False

while True:
    success, frame = cap.read()

    if tracking:
        # Update the tracker
        success, bbox = tracker.update(frame)

        # Draw bounding box
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

    class_label, confidence, prediction = img_classifier(frame)

    # Display class label and confidence
    cv2.putText(frame, f'{class_label} {confidence:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("img", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    # Press 's' to start object tracking
    elif cv2.waitKey(1) == ord('s') and not tracking:
        tracking = True
        bbox = cv2.selectROI("img", frame, False)
        tracker.init(frame, bbox)

cap.release()
cv2.destroyAllWindows()
