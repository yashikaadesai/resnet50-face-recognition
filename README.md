### ResNet50 Face Recognition
Celebrity face recognition system using transfer learning with ResNet50 and Leave-One-Out Cross-Validation (LOOCV).
#### Overview
This project implements a deep learning model to classify 17 different celebrities using facial recognition.

##### Key Feautres
- Pre-trained ResNet50 architecture with transfer learning
- 10-fold Leave-One-Out Cross-Validation
- Data augmentation for improved robustness
- Comprehensive confusion matrix analysis

#### Results
Overall Accuracy: 55.94%

#### Confusion Matrix
<img width="548" height="461" alt="Screenshot 2026-01-05 at 10 25 00 PM" src="https://github.com/user-attachments/assets/460300e7-a5ce-443d-a64c-622c20159642" />

#### Best Performing Celebrities

- Johnny Depp: 68%
- Leonardo DiCaprio: 65%
- Hugh Jackman: 65%

#### Most Confused Pairs

- Robert Downey Jr → Johnny Depp (25 times)
- Megan Fox → Sandra Bullock (18 times)
- Denzel Washington → Will Smith (17 times)

#### Dataset
##### Celebrity Faces Dataset

- 17 celebrity classes
- 100 images per celebrity
- Total: 1,700 images

Each fold contains 170 test images (10 per celebrity) and 1,530 training images.

```bash
# Clone the repository
git clone https://github.com/yashikaadesai/resnet50-face-recognition.git
cd resnet50-face-recognition

# Install dependencies
pip install -r requirements.txt

```

The script will:

1. Load and split data into 10 folds
2. Train ResNet50 on 9 folds and test on 1 fold
3. Repeat for all 10 folds (LOOCV)
4. Generate confusion matrix and results

#### Model Architecture
ResNet50 with Transfer Learning

- Base: Pre-trained on ImageNet
- Modified final layer: 1000 → 17 classes
- Optimizer: Adam (lr=0.001)
- Training: 10 epochs per fold
- Batch size: 32

#### Data Augmentation:

- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness & contrast)
- ImageNet normalization
