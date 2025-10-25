# yolo_inference.py
from ultralytics import YOLO
import cv2


def load_yolo(model_path):
    model = YOLO(model_path)
    return model


def detect_faces(yolo_model, frame, conf_thresh=0.3):
# yolo_model(frame) returns results object list — using .boxes
    results = yolo_model(frame)
    faces = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        faces.append((x1, y1, x2, y2, conf))
    return faces