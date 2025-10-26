# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from yolo_interface import load_yolo, detect_faces
from data_preprocessing_fer import EMOTION_LABELS

st.set_page_config(page_title='Live Emotion Detector', layout='wide')
@st.cache_resource
def load_models(yolo_path, emotion_path):
    yolo = load_yolo(yolo_path)
    emo = load_model(emotion_path)
    return yolo, emo


yolo_path = st.sidebar.text_input('YOLO model path', 'model/yolov8n-face.pt')
emo_path = st.sidebar.text_input('Emotion model path', 'models/model2.h5')
conf_thresh = st.sidebar.slider('YOLO confidence', 0.1, 1.0, 0.35)

st.title('Live Face Emotion Detection')

col1, col2 = st.columns([3,1])
with col1:
    stframe = st.image([])
with col2:
    st.markdown('### Controls')
    run_button = st.button('Start Webcam')


if run_button:
    yolo, emo = load_models(yolo_path, emo_path)
    cap = cv2.VideoCapture(0)
    placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(yolo, frame, conf_thresh)
        for (x1,y1,x2,y2,conf) in faces:
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(frame.shape[1]-1,x2), min(frame.shape[0]-1,y2)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48,48)).astype('float32')/255.0
            face = np.expand_dims(face, -1)
            face = np.expand_dims(face, 0)
            preds = emo.predict(face, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = EMOTION_LABELS[idx]
            score = float(preds[idx])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)


cap.release()