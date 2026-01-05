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

def perform_loocv(folds, class_names, num_epochs=10, batch_size=32):
    """Perform Leave-One-Out Cross-Validation"""
    num_classes = len(class_names)
    all_preds, all_true = [], []
    fold_accuracies = []
    
    for test_idx in range(10):
        print(f"\n{'='*60}")
        print(f"FOLD {test_idx+1}/10")
        print(f"{'='*60}")
        
        
        train_paths, train_labels = [], []
        for i in range(10):
            if i != test_idx:
                train_paths.extend(folds[i]['image_paths'])
                train_labels.extend(folds[i]['labels'])
        
        test_paths = folds[test_idx]['image_paths']
        test_labels = folds[test_idx]['labels']
        
        print(f"Training: {len(train_paths)} images")
        print(f"Testing: {len(test_paths)} images")
        
        train_loader = DataLoader(
            CelebDataset(train_paths, train_labels, train_transform),
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            CelebDataset(test_paths, test_labels, test_transform),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        model = create_model(num_classes)
        print("\nTraining...")
        model = train(model, train_loader, epochs=num_epochs)
        print("Testing...")
        preds, true = test(model, test_loader)
        
        fold_acc = accuracy_score(true, preds)
        fold_accuracies.append(fold_acc)
        all_preds.extend(preds)
        all_true.extend(true)
        
        print(f"\nFold {test_idx+1} Accuracy: {fold_acc*100:.2f}%")
    
    return all_preds, all_true, fold_accuracies

def analyze_results(all_preds, all_true, class_names):
    """Analyze and visualize results"""
    
    overall_acc = accuracy_score(all_true, all_preds)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    
    cm = confusion_matrix(all_true, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Celebrity Face Recognition', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    # Confusion analysis
    print("\n" + "="*60)
    print("CONFUSION ANALYSIS")
    print("="*60)
    print("\nMost Confused Pairs:")
    print("-" * 60)
    
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confusions.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': cm[i][j]
                })
    
    confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)
    
    for idx, conf in enumerate(confusions[:10], 1):
        print(f"{idx}. {conf['true']} misclassified as {conf['predicted']}: {conf['count']} times")
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for i in range(len(cm)):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{class_names[i]:<30}: {acc:.2f}% ({correct}/{total} correct)")
    
    return cm

