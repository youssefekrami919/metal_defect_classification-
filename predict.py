import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# Define the same CNN architecture as in training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MetalDefectPredictor:
    def __init__(self, model_path='metal_defect_model_notebook.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        
        # Load the model
        self.model = SimpleCNN(num_classes=len(self.classes))
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Please train the model first.")
            self.is_loaded = False
            
        # Define the exact same transform as in training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, confidence_threshold=0.5):
        if not self.is_loaded:
            return {"error": "Model is not loaded."}
            
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
            max_prob, predicted_idx = torch.max(probabilities, 0)
            max_prob_value = max_prob.item()
            
            # Confidence Threshold Logic
            if max_prob_value < confidence_threshold:
                # If confidence is low, we assume there's no major defect (Good Metal)
                result_class = "Good (المعدن ده كويس)"
            else:
                result_class = self.classes[predicted_idx.item()]
                
            return {
                "class": result_class,
                "confidence": max_prob_value,
                "all_probabilities": {cls: prob.item() for cls, prob in zip(self.classes, probabilities)}
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predictor = MetalDefectPredictor()
        result = predictor.predict(img_path)
        print(json.dumps(result, indent=4))
    else:
        print("Usage: python predict.py <image_path>")
