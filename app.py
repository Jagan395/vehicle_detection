from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded and processed images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best_yolo.pt")

@app.route('/')
def home():
    # Provide default values to avoid blank page
    return render_template(
        'index.html',
        prediction_text="",
        input_image=None,
        output_image=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file uploaded
    if 'file' not in request.files:
        return render_template('index.html',
                               prediction_text="No file uploaded.",
                               input_image=None,
                               output_image=None)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html',
                               prediction_text="No file selected.",
                               input_image=None,
                               output_image=None)
    
    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Run YOLO prediction
    results = model.predict(image_path, conf=0.25)[0]

    # Draw bounding boxes and save the result
    im_array = results.plot()[..., ::-1]  # Convert BGR to RGB
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

    # Render the HTML template with results
    return render_template(
        'index.html',
        prediction_text=detection_text,
        input_image=file.filename,
        output_image=output_filename
    )

# Serve uploaded images
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    # Explicit host and port for VS Code
    app.run(host="127.0.0.1", port=5000, debug=True)
