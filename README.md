Live Face Emotion Detection

This project implements a real-time face emotion detection system using OpenCV, YOLO for face detection, and a custom-trained Convolutional Neural Network (CNN) for emotion classification.

The emotion classification model is trained using a two-stage transfer learning approach:

    Initial training on the FER2013 dataset.

    Fine-tuning on the IITM Face Emotion Dataset, incorporating class weights to handle data imbalance and improve prediction normalization.

Features

    Face Detection: Uses a pre-trained YOLO model (specifically a face-detection variant) to locate faces in a video stream.

    Emotion Classification: Employs a CNN trained on facial expression datasets (FER2013 and IITM) to classify detected faces into distinct emotion categories.

    Transfer Learning: Leverages knowledge from the larger FER2013 dataset before specializing on the smaller IITM dataset.

    Class Weighting: Addresses potential bias towards common emotions (like 'Happy'/'Smile') during fine-tuning by applying class weights.

    Live Demo: Provides a real-time demonstration using OpenCV to capture webcam feed, run detection/classification, and display results with emotion probabilities.

    TensorBoard Logging: Training script includes TensorBoard integration for monitoring training progress.

Setup

  1) Clone the Repository:
        git clone <https://github.com/Zomwolff/bhAi>
        cd bhAi
  2) Create Virtual Environment (Recommended):
        python -m venv venv
        # Activate:
        # Windows: venv\Scripts\activate
        # macOS/Linux: source venv/bin/activate
  3) Install Dependencies
        pip install --upgrade pip
        pip install -r requirements.txt
  4) To run the 1st demo
        streamlit run streamlit_app.py
  5) To run the 2nd demo
        python live_demo.py --yolo model/yolov8n-face.pt --emotion_models model/model1.h5 models/model2.h5
