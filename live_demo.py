# File: live_demo_ensemble.py
# Loads multiple emotion models, averages predictions, displays mean probabilities,
# and limits processing to a target FPS.

import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from yolo_interface import load_yolo, detect_faces # Assuming you have yolo_inference.py
import os
import time # For FPS calculation and limiting

# --- Configuration ---
# (Label loading logic remains the same as previous version)
try:
    from data_preprocessing_fer import EMOTION_LABELS # For FER (7 classes)
    EMOTION_LABELS = EMOTION_LABELS
    NUM_CLASSES = len(EMOTION_LABELS)
    print("Using FER2013 Labels (7 classes)")
except ImportError:
    try:
        from run_combined_training_iitm_filenames_weighted import EMOTION_LABELS_IITM # For IITM (6 classes)
        EMOTION_LABELS = EMOTION_LABELS_IITM
        NUM_CLASSES = len(EMOTION_LABELS)
        print("Using IITM Labels (6 classes)")
    except ImportError:
        print("ERROR: Could not import emotion label dictionaries. Please ensure data_preprocessing_fer.py or the training script exists.")
        EMOTION_LABELS = {0: 'Class0', 1: 'Class1', 2: 'Class2', 3: 'Class3', 4: 'Class4', 5: 'Class5', 6: 'Class6'} # Fallback
        NUM_CLASSES = 7
        print("Warning: Using generic fallback labels.")


INPUT_SHAPE = (48, 48, 1) # Expected input shape for emotion models

# Sidebar settings (remain the same)
SIDEBAR_WIDTH = 200
SIDEBAR_BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
LINE_HEIGHT = 20
SIDEBAR_START_X = -1
SIDEBAR_START_Y = 10

# --- Helper Functions (prepare_face, draw_sidebar remain the same) ---
def prepare_face(face_img):
    """Preprocesses a single face image for the emotion CNN."""
    try:
        face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (INPUT_SHAPE[1], INPUT_SHAPE[0])) # Use INPUT_SHAPE constants
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, -1) # Add channel dim -> (48, 48, 1)
        face = np.expand_dims(face, 0)  # Add batch dim -> (1, 48, 48, 1)
        if face.shape != (1,) + INPUT_SHAPE:
             print(f"Warning: Prepared face shape mismatch. Expected {(1,) + INPUT_SHAPE}, got {face.shape}")
             return None
        return face
    except Exception as e:
        print(f"Error preparing face: {e}")
        return None

def draw_sidebar(frame, probabilities, labels_dict):
    """Draws the emotion probabilities on the side of the frame."""
    global SIDEBAR_START_X # Use global to set it once
    if SIDEBAR_START_X == -1:
        # Calculate sidebar start X based on current frame width
        if frame.shape[1] > SIDEBAR_WIDTH:
             SIDEBAR_START_X = frame.shape[1] - SIDEBAR_WIDTH
        else:
             SIDEBAR_START_X = 0 # Handle cases where frame is narrower than sidebar

    # Ensure sidebar start doesn't go negative if calculated strangely
    sidebar_x = max(0, SIDEBAR_START_X)

    # Draw sidebar background
    cv2.rectangle(frame, (sidebar_x, 0), (frame.shape[1], frame.shape[0]), SIDEBAR_BG_COLOR, -1)

    y_pos = SIDEBAR_START_Y

    if probabilities is not None and len(probabilities) == len(labels_dict):
        sorted_labels = sorted(labels_dict.items(), key=lambda item: item[1]) # Sort by label name
        max_prob_idx = np.argmax(probabilities)

        for index, label in sorted_labels:
            if index < len(probabilities): # Safety check
                prob = probabilities[index]
                text = f"{label}: {prob:.2f}"
                color = (0, 255, 0) if index == max_prob_idx else TEXT_COLOR # Highlight max prob
                cv2.putText(frame, text, (sidebar_x + 10, y_pos), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)
                y_pos += LINE_HEIGHT
            else:
                 print(f"Warning: Index {index} out of bounds for probabilities array (length {len(probabilities)})")

    else:
        cv2.putText(frame, "No face detected", (sidebar_x + 10, y_pos), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


# --- Main Function ---
def main(args):
    global SIDEBAR_START_X
    SIDEBAR_START_X = -1 # Reset sidebar position calculation

    # --- Load Models (YOLO and Emotion Ensemble - same as before) ---
    # (YOLO loading code...)
    if not os.path.exists(args.yolo):
        print(f"❌ Error: YOLO model file not found at '{args.yolo}'")
        return
    try:
        print(f"Loading YOLO model: {args.yolo}...")
        yolo_model = load_yolo(args.yolo)
        print("✅ YOLO model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    # (Emotion model loading loop - same as before)
    emotion_models = []
    print("\nLoading Emotion models:")
    if not args.emotion_models:
        print("❌ Error: No emotion model paths provided via --emotion_models argument.")
        return
    for model_path in args.emotion_models:
        # (Error checking and loading logic for each emotion model...)
        if not os.path.exists(model_path):
            print(f"⚠️ WARNING: Emotion model file not found at '{model_path}'. Skipping.")
            continue
        try:
            print(f" - Loading: {model_path}...")
            model = load_model(model_path)
            # Basic checks
            model_input_shape = tuple(model.input.shape[1:])
            model_output_classes = model.output.shape[-1]
            if model_input_shape != INPUT_SHAPE: print(f"    ⚠️ WARNING: Model '{os.path.basename(model_path)}' input shape {model_input_shape} != expected {INPUT_SHAPE}.")
            if model_output_classes != NUM_CLASSES: print(f"    ⚠️ WARNING: Model '{os.path.basename(model_path)}' output classes {model_output_classes} != expected {NUM_CLASSES}.")

            emotion_models.append(model)
            print(f"   ✅ Loaded successfully.")
        except Exception as e:
            print(f"   ❌ Error loading emotion model '{model_path}': {e}. Skipping.")

    if not emotion_models:
        print("\n❌ Error: No valid emotion models were loaded. Exiting.")
        return

    print(f"\n✅ Successfully loaded {len(emotion_models)} emotion model(s).")
    print(f"--- Processing at approx. {args.target_fps} FPS ---")
    print("--- Starting Webcam ---")

    # --- Start Webcam ---
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera index {args.cam}")
        return

    # --- FPS Limiting Variables ---
    process_interval = 1.0 / args.target_fps # Time needed between processing frames
    last_process_time = time.time()
    last_mean_preds = None # Store preds from last processed frame
    frame_count_display = 0 # For calculating display FPS
    start_time_display = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        current_time = time.time()
        frame_count_display += 1
        processed_this_frame = False # Flag to know if we updated predictions

        # --- Check if enough time has passed to process this frame ---
        if (current_time - last_process_time) >= process_interval:
            last_process_time = current_time # Reset timer
            processed_this_frame = True # Mark that we are processing

            current_preds_list = [] # Store predictions for faces in THIS frame

            # --- 4. Detect Faces ---
            try:
                faces = detect_faces(yolo_model, frame, conf_thresh=args.conf)
            except Exception as yolo_e:
                print(f"Error during YOLO detection: {yolo_e}")
                faces = []

            # --- 5. Process Each Face ---
            for (x1, y1, x2, y2, conf) in faces:
                # (Coordinate validation - same as before)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                if x1 >= x2 or y1 >= y2: continue

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0: continue

                inp = prepare_face(face_img)
                if inp is None: continue

                # --- 6. Get Predictions & 7. Average ---
                all_preds = []
                try:
                    for em_model in emotion_models:
                        preds = em_model.predict(inp, verbose=0)[0]
                        if len(preds) == NUM_CLASSES:
                            all_preds.append(preds)
                        # else: warning printed during loading

                except Exception as pred_e:
                    print(f"Error during emotion prediction: {pred_e}")
                    continue # Skip face

                if all_preds:
                    mean_preds = np.mean(np.array(all_preds), axis=0)
                    current_preds_list.append(mean_preds)
                    # Don't draw box here, draw all boxes after processing check

            # Update last_mean_preds if faces were found and processed in this frame
            if current_preds_list:
                last_mean_preds = current_preds_list[-1]
            elif not faces: # If no faces were detected in this processed frame
                 last_mean_preds = None # Clear old predictions

        # --- END OF FPS LIMITED BLOCK ---

        # --- Draw ALL detected face boxes from the *last processed frame*
        # We need to store faces from the last processed frame to redraw them
        # Re-detecting here for simplicity, but could store them from the IF block
        try:
             # Re-detect faces just for drawing boxes (quick operation)
             faces_for_drawing = detect_faces(yolo_model, frame, conf_thresh=args.conf)
             for (x1, y1, x2, y2, conf) in faces_for_drawing:
                 x1, y1 = max(0, x1), max(0, y1)
                 x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                 if x1 < x2 and y1 < y2:
                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as draw_e:
             print(f"Error detecting faces for drawing: {draw_e}")


        # --- 9. Draw Sidebar (Always do this) ---
        # It uses last_mean_preds, which is only updated when processed_this_frame is True
        draw_sidebar(frame, last_mean_preds, EMOTION_LABELS)

        # --- Calculate and Display *Display* FPS ---
        end_time_display = time.time()
        elapsed_time_display = end_time_display - start_time_display
        if elapsed_time_display > 0:
            fps_display = frame_count_display / elapsed_time_display
            cv2.putText(frame, f"Display FPS: {fps_display:.1f}", (10, 30), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f" Target Proc FPS: {args.target_fps}", (10, 55), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # --- 10. Display Frame (Always do this) ---
        cv2.imshow('Ensemble Emotion Detector (Limited FPS)', frame)

        # --- 11. Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Webcam stopped ---")

# --- Argument Parser ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live face emotion detection using an ensemble of models with FPS limiting.")
    parser.add_argument('--yolo', required=True, help="Path to the YOLO face detection model (.pt)")
    parser.add_argument('--emotion_models', required=True, nargs='+', help="Paths to the trained Keras emotion classification models (.h5)")
    parser.add_argument('--cam', type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument('--conf', type=float, default=0.35, help="YOLO confidence threshold (default: 0.35)")
    # New argument for FPS limiting
    parser.add_argument('--target_fps', type=float, default=3.0, help="Target processing FPS (e.g., 2.0, 3.0, 4.0)")

    args = parser.parse_args()

    # --- Check TensorFlow GPU (same as before) ---
    print("TensorFlow Version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices: print(f"--- Using GPU: {gpu_devices[0].name} ---")
    else: print("--- No GPU detected by TensorFlow, using CPU ---")

    main(args)
