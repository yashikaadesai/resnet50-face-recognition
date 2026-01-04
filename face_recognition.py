import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

#Data Preparation

def load_and_split_data(data_dir):
    """Load celebrity face dataset and split into 10 folds"""
    data_path = Path(data_dir)
    
    celebrity_folders = [f for f in sorted(data_path.iterdir()) 
                        if f.is_dir() and not f.name.startswith('.')]
    
    print(f"Found {len(celebrity_folders)} celebrity classes:")
    print("-" * 60)
    
    # Initialize 10 empty folds
    folds = [{'image_paths': [], 'labels': []} for _ in range(10)]
    class_names = []
    
    for class_idx, celeb_folder in enumerate(celebrity_folders):
        celeb_name = celeb_folder.name
        class_names.append(celeb_name)
        
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            all_images.extend(list(celeb_folder.glob(f'*{ext}')))
        
        all_images = sorted(all_images)
        
        print(f"Class {class_idx}: {celeb_name:<30} - {len(all_images)} images")
        
        images_per_fold = len(all_images) // 10
        
        for fold_idx in range(10):
            start_idx = fold_idx * images_per_fold
            end_idx = start_idx + images_per_fold
            fold_images = all_images[start_idx:end_idx]
            
            folds[fold_idx]['image_paths'].extend([str(img) for img in fold_images])
            folds[fold_idx]['labels'].extend([class_idx] * len(fold_images))
    
    # Verify folds
    print("\n" + "="*60)
    print("FOLD SUMMARY")
    print("="*60)
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: {len(fold['image_paths'])} images")
    
    return folds, class_names

class CelebDataset(Dataset):
    """Dataset class for celebrity faces"""
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_model(num_classes):
    """Create ResNet50 pre-trained model"""
    print("Loading pre-trained ResNet50...")
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train(model, loader, epochs=10):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            acc = 100. * correct / total
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}, Acc: {acc:.2f}%')
    
    return model

def test(model, loader):
    """Test the model"""
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(1).cpu().numpy())
            true.extend(labels.numpy())
    return preds, true
