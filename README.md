# Metal Surface Defect Classification

This project uses a Deep Learning (CNN) model to identify defects on metal surfaces. If no defect is clearly identified, the system classifies the surface as "Good".

## Features
- **Centralized System**: Run `main.py` to handle everything from training to launching the Web GUI.
- **Classification**: Detects 6 types of defects (Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches).
- **"Good" Detection**: Automatically identifies a surface as "Good" (المعدن ده كويس) if the defect confidence is below 50%.
- **Premium Web GUI**: Interactive dark-mode interface with drag-and-drop support.

## Project Structure
- `main.py`: The main entry point. Prepares the model and starts the web server.
- `app.py`: Flask application logic.
- `train_logic.py`: Contains the CNN architecture and training logic.
- `train_model.ipynb`: Jupyter version of the training process (optional).
- `test_model.ipynb`: Notebook for manual image testing.
- `predict.py`: core prediction functions.
- `documentation.txt` & `report.txt`: Project documentation and results.

## Running Steps

### 1. Preparation
Ensure your dataset is located at:
`E:\Semester 8\Advanced Methods\Project\data\NEU-DET`

### 2. Launching the System
Simply run the following command in your terminal:
```bash
python main.py
```

**What happens next?**
1. The script checks for the existence of `metal_defect_model_notebook.pkl`.
2. If the file is missing, it will automatically start the training process and show a progress bar.
3. Once the model is ready, it will launch the **Web GUI**.
4. Open your browser at `http://127.0.0.1:5000` to start inspecting metal photos.

## Requirements
- Python 3.x
- PyTorch & Torchvision
- Flask
- PIL (Pillow)
- tqdm
- Jupyter Notebook
