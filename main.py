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

# --- Session Statistics ---
session_stats = {
    "total_inspections": 0,
    "defects_found": 0,
    "avg_confidence": 0.0,
    "defect_counts": {
        "crazing": 0,
        "inclusion": 0,
        "patches": 0,
        "pitted_surface": 0,
        "rolled-in_scale": 0,
        "scratches": 0,
        "Good": 0
    }
}

# --- Video Tracking ---
latest_video_status = {"class": "Waiting for data...", "confidence": 0.0}

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
        
        # Update Stats
        global session_stats
        session_stats["total_inspections"] += 1
        conf = result.get("confidence", 0)
        session_stats["avg_confidence"] = (session_stats["avg_confidence"] * (session_stats["total_inspections"] - 1) + conf) / session_stats["total_inspections"]
        
        pred_class = result.get("class", "Good")
        clean_class = "Good" if "Good" in pred_class else pred_class
        if clean_class in session_stats["defect_counts"]:
            session_stats["defect_counts"][clean_class] += 1
            if clean_class != "Good":
                session_stats["defects_found"] += 1
        
        return jsonify(result)

@app.route('/dashboard_stats')
def dashboard_stats():
    return jsonify(session_stats)

@app.route('/training_info')
def training_info():
    try:
        import json
        with open('training_history.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"error": "Training history not found"}), 404

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
        
        # Update global status instead of drawing on frame
        global latest_video_status
        latest_video_status = {
            "class": label,
            "confidence": round(max_prob.item(), 4)
        }
        
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

@app.route('/video_status')
def video_status():
    return jsonify(latest_video_status)

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
