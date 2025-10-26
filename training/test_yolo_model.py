# File: test_yolo_model.py

import argparse
from ultralytics import YOLO
import cv2
import os

def test_model(model_path, image_path, output_dir="output_tests"):
    """
    Loads a YOLO model, runs inference on an image, and saves the result.
    """
    # --- Input Validation ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Loading model: {model_path}")
    print(f"Testing on image: {image_path}")

    try:
        # --- Load Model ---
        model = YOLO(model_path) #

        # --- Perform Inference ---
        # The model automatically handles reading the image
        results = model.predict(source=image_path, save=False, show=False) #

        # --- Process Results ---
        # results is a list (usually with one item for one image)
        if results and len(results) > 0:
            # Get the first result object
            result = results[0]

            # The 'plot()' method returns the image with bounding boxes drawn
            annotated_image = result.plot()

            # --- Save Output ---
            os.makedirs(output_dir, exist_ok=True)
            # Create a unique output filename based on model and image names
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{model_name}_on_{image_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(output_path, annotated_image)
            print(f"✅ Test successful! Output saved to: {output_path}")

            # Optional: Display the image in a window
            # cv2.imshow("YOLO Test Result", annotated_image)
            # cv2.waitKey(0) # Wait for a key press
            # cv2.destroyAllWindows()

        else:
            print("Model ran but produced no results.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a YOLOv8/v11 .pt model")
    parser.add_argument("--model", required=True, help="Path to the .pt model file (e.g., yolov8n.pt)")
    parser.add_argument("--image", required=True, help="Path to the image file to test on")
    parser.add_argument("--output", default="output_tests", help="Directory to save the annotated image")

    args = parser.parse_args()
    test_model(args.model, args.image, args.output)