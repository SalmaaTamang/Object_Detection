# import cv2
# import numpy as np
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

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load ASL model
model = load_model('mnist_model.h5')

# Load class labels
with open('labels.names', 'r') as f:
    labels = f.read().split('\n')

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Assuming your ASL model doesn't require hand landmarks, you can use the entire frame
    # (You might need to resize or preprocess the frame depending on your model's input requirements)
    input_data=cv2.resize(frame_rgb,(224,224))
    input_data = np.expand_dims(input_data, axis=0)

    # Make prediction using the ASL model
    prediction = model.predict(input_data)
    print(prediction)
    class_id = np.argmax(prediction)
    class_name = labels[class_id]

    # Display the result on the frame
    cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Mnist Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
