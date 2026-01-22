from ultralytics import YOLO
from PIL import Image
import os, shutil, base64
from io import BytesIO

model = YOLO("best_yolo.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detection(file):
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(image_path, conf=0.25)[0]

    im_array = results.plot()[..., ::-1]
    image = Image.fromarray(im_array)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        detections.append(f"{cls_name} ({conf:.2f})")

    return {
        "image": encoded_image,
        "detections": detections
    }
