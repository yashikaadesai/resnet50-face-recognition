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
