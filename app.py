import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
from ultralytics import YOLO
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

model = YOLO('best (4).pt')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server's Running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg','.jfif')):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        image = Image.open(io.BytesIO(image_file.read()))
        image = image.convert("RGB")  # تحويل الصورة إلى RGB

        # تصغير الصورة مع الحفاظ على التناسب
        max_size = (640, 640)
        image.thumbnail(max_size, Image.LANCZOS)

        results = model.predict(image)
        predictions = []

        if not results:
            return jsonify({[]}), 200

        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = result.xyxy.tolist()[0]
            class_id = int(result.cls)
            confidence = float(result.conf)
            class_name = model.names[class_id]

            predictions.append({
                #'x_min': x_min,
                #'y_min': y_min,
                #'x_max': x_max,
                #'y_max': y_max,
                '' class_name '',
                'confidence': confidence
            })

        return jsonify({'predictions': predictions}), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
