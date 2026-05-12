import os
import sys
from train_logic import run_training
from app import app

MODEL_NAME = 'metal_defect_model_notebook.pkl'

def start_project():
    print("========================================")
    print("   Metal Defect Inspection AI System")
    print("========================================")

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

    # 2. Start the Web GUI
    print("\n[*] Starting Web GUI on http://127.0.0.1:5000")
    print("[*] You can now open your browser and drag & drop images.")
    print("========================================\n")
    
    # Run the Flask app
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    start_project()
