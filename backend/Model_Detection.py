from ultralytics import YOLO
from PIL import Image
import os
import shutil

# Load model once (important for performance)
model = YOLO("backend/best_yolo.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def detection(file):
    # Save uploaded image (FastAPI way)
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model.predict(image_path, conf=0.25)[0]

    # Draw bounding boxes
    im_array = results.plot()[..., ::-1]  # BGR → RGB
    output_filename = f"detected_{file.filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    Image.fromarray(im_array).save(output_path)

    # Extract detection details
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        detections.append(f"{cls_name} ({conf:.2f})")

    detection_text = ", ".join(detections) if detections else "No objects detected."

    return {
        "original_image": image_path,
        "output_image": output_path,
        "detections": detection_text
    }
