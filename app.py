from flask import Flask, request, render_template
from ultralytics import YOLO
import pandas as pd
import os
from PIL import Image
import cv2

app = Flask(__name__)

# Folder configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model_path = 'yolov8n.pt'  # Replace with the path to your trained YOLOv8 model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = YOLO(model_path)

# Read accuracy metrics from CSV
csv_path = 'results.csv'
avg_mAP50, avg_mAP50_95 = None, None
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        avg_mAP50 = round(df['metrics/mAP50(B)'].mean(), 4)
        avg_mAP50_95 = round(df['metrics/mAP50-95(B)'].mean(), 4)
    except Exception as e:
        print(f"Error reading metrics from CSV: {e}")
else:
    print("results.csv not found. Metrics will not be displayed.")

@app.route('/')
def index():
    return render_template('index.html', 
                           mAP50=avg_mAP50 if avg_mAP50 else "N/A", 
                           mAP50_95=avg_mAP50_95 if avg_mAP50_95 else "N/A")

@app.route('/upload', methods=['POST'])
def upload():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded!", 
                               mAP50=avg_mAP50, mAP50_95=avg_mAP50_95)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected!", 
                               mAP50=avg_mAP50, mAP50_95=avg_mAP50_95)

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Perform inference with YOLOv8
        results = model(filepath)

        # Check if predictions exist
        if len(results[0].boxes) == 0:
            return render_template('index.html', 
                                   uploaded_image=file.filename, 
                                   processed_image=None, 
                                   error="No objects detected!", 
                                   mAP50=avg_mAP50, 
                                   mAP50_95=avg_mAP50_95)

        # Get the predictions
        predictions = results[0].boxes

        # Annotate the image with labels and confidence
        img = cv2.imread(filepath)  # Read the image with OpenCV
        for prediction in predictions:
            # Get bounding box, class label, and confidence score
            x1, y1, x2, y2 = map(int, prediction.xyxy[0])  # Bounding box coordinates
            conf = round(prediction.conf[0].item(), 2)  # Confidence score
            label = "Elephant"  # Label for the object

            # Draw the bounding box and add label with confidence score
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the annotated image
        output_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        cv2.imwrite(output_path, img)

        # Ensure the processed image file path is valid
        processed_image = file.filename if os.path.exists(output_path) else None

        # Render results
        return render_template('index.html', 
                               uploaded_image=file.filename, 
                               processed_image=processed_image, 
                               mAP50=avg_mAP50, 
                               mAP50_95=avg_mAP50_95)
    except Exception as e:
        return render_template('index.html', 
                               error=f"Error during inference: {e}", 
                               mAP50=avg_mAP50, 
                               mAP50_95=avg_mAP50_95)

if __name__ == '__main__':
    app.run(debug=True)
