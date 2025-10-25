# live_demo.py
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from yolo_inference import load_yolo, detect_faces
from data_preprocessing import EMOTION_LABELS


def prepare_face(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48,48))
    face = face.astype('float32')/255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face


def main(args):
    yolo = load_yolo(args.yolo)
    emotion_model = load_model(args.emotion)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Cannot open camera')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(yolo, frame, conf_thresh=args.conf)
        for (x1,y1,x2,y2,conf) in faces:
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(frame.shape[1]-1,x2), min(frame.shape[0]-1,y2)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            inp = prepare_face(face_img)
            preds = emotion_model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = EMOTION_LABELS[idx]
            score = float(preds[idx])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            text = f'{label} {score:.2f}'
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', required=True)
    parser.add_argument('--emotion', required=True)
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.35)
    args = parser.parse_args()
    main(args)