import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetalDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, 'images', '*.jpg'))
        self.transform = transform
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        if "rolled-in_scale" in filename: label_name = "rolled-in_scale"
        elif "pitted_surface" in filename: label_name = "pitted_surface"
        else: label_name = filename.split('_')[0]
        label = self.class_to_idx[label_name]
        if self.transform: image = self.transform(image)
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

def run_training(epochs=5):
    print("--- Starting Model Training ---")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = r'E:\Semester 8\Advanced Methods\Project\data\NEU-DET\train'
    val_dir = r'E:\Semester 8\Advanced Methods\Project\data\NEU-DET\validation'
    
    if not os.path.exists(train_dir):
        print(f"Error: Dataset not found at {train_dir}")
        return False

    train_dataset = MetalDefectDataset(train_dir, transform=transform)
    val_dataset = MetalDefectDataset(val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f'Validation Accuracy: {val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'metal_defect_model_notebook.pkl')
            print("Model saved to metal_defect_model_notebook.pkl")
    
    print("--- Training Completed ---")
    return True
