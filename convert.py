import os
from PIL import Image

# Paths
wider_img_dir = "dataset/images/val"       # Make sure this path is correct
wider_anno_file = "wider_face_val_bbx_gt.txt" # Make sure this path is correct
yolo_label_dir = "dataset/labels/val"

os.makedirs(yolo_label_dir, exist_ok=True)

with open(wider_anno_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    img_name = lines[i].strip()
    i += 1
    
    # Handle potential non-integer lines if file is malformed
    try:
        num_faces = int(lines[i].strip())
    except ValueError:
        # Skip this entry if the face count is not a number
        print(f"Skipping entry for {img_name}: invalid face count.")
        i += 1 
        continue
        
    i += 1

    # Open image to get dimensions
    img_path = os.path.join(wider_img_dir, img_name)
    if not os.path.exists(img_path):
        # If image doesn't exist, skip its annotation lines
        print(f"Warning: Image not found {img_path}. Skipping {num_faces} faces.")
        i += num_faces
        continue
        
    try:
        img = Image.open(img_path)
        W, H = img.size
    except Exception as e:
        # Handle corrupted images
        print(f"Error opening image {img_path}: {e}. Skipping {num_faces} faces.")
        i += num_faces
        continue

    # Create YOLO label file
    label_path = os.path.join(yolo_label_dir, img_name.replace(".jpg", ".txt"))
    # Ensure the subdirectory (e.g., "0--Parade") exists
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    # FIX 1: Variable name typo. 'labelpath' should be 'label_path'
    with open(label_path, "w") as lf:
        
        # FIX 2: Syntax error. Added '_' as the loop variable
        for _ in range(num_faces):
            parts = lines[i].strip().split()
            i += 1
            
            # WIDER format: [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose]
            # We only need the first 4, but we should check the 'invalid' flag (index 7)
            
            # Basic check for a valid annotation line
            if len(parts) < 10:
                print(f"Warning: Corrupt annotation line for {img_name}. Skipping line.")
                continue

            invalid = int(parts[7])
            # If the face is marked as 'invalid' (e.g., too small), skip it
            if invalid == 1:
                continue

            x, y, w, h = map(float, parts[:4])

            # Skip boxes with zero or negative width/height
            if w <= 0 or h <= 0:
                continue

            # Convert to YOLO format
            x_center = (x + w/2) / W
            y_center = (y + h/2) / H
            w_norm = w / W
            h_norm = h / H
            
            # Write to file
            lf.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete.")