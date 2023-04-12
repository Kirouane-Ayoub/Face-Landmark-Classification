
import mediapipe as mp 
import cv2 
import numpy as np 
from joblib import load
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
model = load("KNN.joblib")
with mp_holistic.Holistic(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as holistic :
    while True:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # recolor feed BGR to RGB 
        image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        # MAKE detections
        result = holistic.process(image)
        # U can print result (landmarks)
        # Recolor our image frome RGB to BGR
        image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
        # draw face landmarks 
        mp_drawing.draw_landmarks(image , result.face_landmarks , mp_holistic.FACEMESH_CONTOURS , 
                                  # for landmarks
                                 mp_drawing.DrawingSpec(color=(255,0,0) ,thickness=1 , circle_radius = 0) ,
                                  # for connections
                                 mp_drawing.DrawingSpec(color=(0,0,255) ,thickness=1 , circle_radius = 0) 
                                 )
        
        try : 
            # extracy face landmark 
            face = result.face_landmarks.landmark
            face_row = list(np.array([[landmark.x , landmark.y , landmark.z] for landmark in face ]).flatten())
            x = pd.DataFrame([face_row])
            classf = model.predict(x)[0]
            cv2.putText(image,classf, (200, 465), cv2.FONT_HERSHEY_DUPLEX , 1, (255, 255, 255), 2)
            cv2.rectangle(image, (117,424), (548, 479), (255, 255, 255), 1)
        except: 
            pass
        # Show the captured image
        cv2.imshow('WebCam', image)

        # wait for the key and come out of the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Discussed below
cap.release()
cv2.destroyAllWindows()