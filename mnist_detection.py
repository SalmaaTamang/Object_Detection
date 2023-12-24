import cv2
import numpy as np
from tensorflow.keras.models import load_model

model=load_model('mnist_model.h5')

classes={0:'6',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}

def img_classifier(img):

  img=cv2.resize(img,(224,224))
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  img=img.reshape(1,224,224,3)
  img=img/255.0

  predication=model.predict(img)
  class_index=np.argmax(predication)
  class_label=classes[class_index]
  confidence=predication[0][class_index]
  return  class_label,confidence

cap=cv2.VideoCapture(0)
while True:
  success,frame=cap.read(0)

  if not success:
    break
  class_label,confidence=img_classifier(frame)  
  cv2.putText(frame, f'Digit: {class_label} - Confidence: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('Video', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()        
