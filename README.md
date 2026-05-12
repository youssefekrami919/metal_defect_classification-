# Metal Surface Defect Classification System

This project uses a Deep Learning (CNN) model to identify defects on metal surfaces in both static images and real-time video streams.

## Features
- **All-in-One Launcher**: Run `main.py` to train the model (if needed) and launch the Web GUI.
- **Image & Video Analysis**: Dedicated tabs for analyzing static photos and inspection videos.
- **"Good" Detection**: Automatically identifies a surface as "Good" (المعدن ده كويس) if the defect confidence is below 50%.
- **Premium Interface**: Modern dark-mode interface with drag-and-drop and real-time video overlays.

## Project Structure
- `main.py`: The main entry point (Server + Logic).
- `predict.py`: Core prediction class and model definitions.
- `train_logic.py`: Training pipeline and dataset handling.
- `test_model.ipynb`: Jupyter Notebook for manual testing of specific images.
- `documentation.txt` & `report.txt`: Technical details and project results.
- `static/` & `templates/`: Web interface assets.

## Running the System

### 1. Preparation
Ensure your dataset is located at:
`E:\Semester 8\Advanced Methods\Project\data\NEU-DET`

### 2. Launching
Run the following command:
```bash
python main.py
```
1. If `metal_defect_model_notebook.pkl` is missing, the system will train itself first.
2. The Web GUI will launch at `http://127.0.0.1:5000`.

## Requirements
- Python 3.x
- PyTorch & Torchvision
- Flask
- OpenCV (`opencv-python`)
- PIL (Pillow)
- tqdm
