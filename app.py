import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time
import pandas as pd
from joblib import load
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with st.sidebar:
    st.image("icon.png")
    detectf = st.selectbox("Detect from : " , ["File" , "webCam" , "URL"])
    save = st.radio("Do you want to save Results ? " , ["Yes" , "No"])
    class_model = st.selectbox("Select the Classification Model : " , ["KNN" , "LogisticRegression" , "MLPClassifier" , "SVC"])
    

model = load(f"{class_model}.joblib")
tab0 , tab1 = st.tabs(["HOME" , "DETECTION"])
with tab0 : 
    st.header("About This Project : ")
    st.image("home.jpg")
    st.write(""" 
        The Emotions Detection using Face Landmark Classification project is aimed at developing a system that
        can accurately detect and classify emotions from facial landmarks. It involves analyzing facial features
        such as eyes, eyebrows, nose, mouth, and jawline to determine the emotional state of an individual.
        The system uses machine learning techniques, such as convolutional neural networks (CNNs) or deep learning models, 
        trained on a large dataset of labeled facial landmark images. The project aims to achieve high accuracy in emotion detection, 
        including emotions such as happiness, sadness, anger, surprise, fear, and disgust. The system can be
        integrated into various applications, such as facial emotion recognition in human-computer interaction, 
        virtual reality, gaming, and mental health assessment. The project may also involve data preprocessing, 
        feature extraction, model training and evaluation, and optimization for real-time or embedded systems.
        Ethical considerations, such as privacy and bias, will also be taken into account in the project's implementation.
    """)

with tab1 :
    if detectf == "File" : 
        file_ = st.file_uploader("Upload your Video : " , type=["mp4" , "mkv", "webm"])
        if file_ : 
            source = file_.name
    elif detectf == "webCam" : 
        source = st.selectbox("Select Your Webcam Index : " , (1 , 2 , 3))
    elif detectf == "URL" : 
        source = st.text_input("Input Your URL here and click Entre : ")
    col1 , col2 , col3 = st.columns(3)
    with col1 : 
        st.write("Click To Start Detection : ")
        startb = st.button("Start") 
    with col3 :
        if save == "Yes" : 
            st.write("Click To Save Results : ")
        else : 
            st.write("Click To Stop")
        stop_saveb = st.button("Stop")
    # Set mediapipe model 
    if startb :
        cap = cv2.VideoCapture(source)
        if save == "Yes" :  
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
            out = cv2.VideoWriter(f'results/{str(time.asctime())}.mp4',
                                fourcc, 10, (w, h)) 
        else : 
            pass
        frame_window = st.image([])
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
                    cv2.putText(image,classf, (50, 50), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 255, 255), 2)
                except: 
                    pass
                img = cv2.cvtColor( image , cv2.COLOR_BGR2RGB)
                frame_window.image(img)
                try : 
                    out.write(image) 
                except : 
                    pass
                if stop_saveb : 
                    cap.release()
                    out.release()