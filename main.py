import os
import sys
import cv2
import torch
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from PIL import Image
from train_logic import run_training, SimpleCNN
from predict import MetalDefectPredictor

# --- Configuration ---
MODEL_NAME = 'metal_defect_model_notebook.pkl'
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # 64MB

# Global predictor (will be initialized in start_project)
predictor = None

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = predictor.predict(filepath)
        return jsonify(result)

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        transform = predictor.transform
        input_tensor = transform(pil_img).unsqueeze(0).to(predictor.device)
        
        with torch.no_grad():
            outputs = predictor.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        max_prob, predicted_idx = torch.max(probabilities, 0)
        label = predictor.classes[predicted_idx.item()] if max_prob.item() >= 0.5 else "Good (المعدن ده كويس)"
        
        color = (0, 255, 0) if "Good" in label else (0, 0, 255)
        cv2.putText(frame, f"Result: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Conf: {max_prob.item():.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    video_filename = request.args.get('filename')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename})
    return jsonify({'error': 'Upload failed'}), 400

# --- Entry Point ---
def start_project():
    global predictor
    print("========================================")
    print("   Metal Defect Inspection AI System")
    print("========================================")

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # 1. Check if model exists
    if not os.path.exists(MODEL_NAME):
        print(f"\n[!] Model '{MODEL_NAME}' not found.")
        print("[*] Starting initial training to prepare the model...")
        success = run_training(epochs=5)
        if not success:
            print("\n[!] Training failed. Please check your dataset path.")
            return
    else:
        print(f"\n[+] Model '{MODEL_NAME}' is ready.")

    # 2. Initialize Predictor
    predictor = MetalDefectPredictor(MODEL_NAME)

    # 3. Start the Web GUI
    print("\n[*] Starting Web GUI on http://127.0.0.1:5000")
    print("[*] You can now open your browser and analyze images/videos.")
    print("========================================\n")
    
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    start_project()
