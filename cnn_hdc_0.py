"""
Hybrid CNN-HDC Wildfire Detection System (Optimized)
Authors: Ada Qi (PID: A16999495)
ECE 176 Project - University of California San Diego
"""

import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchhd import embeddings, functional, models as hd_models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
class Config:
    # Dataset paths
    train_root = "data/train"
    test_root = "data/test"
    
    # Model parameters
    cnn_feature_dim = 512  # ResNet18 feature dimension
    hdc_dim = 4096         # Hypervector dimension
    quantize = "binary"    # "binary" or "ternary"
    
    # Training parameters
    batch_size = 32   # Increased batch size for efficiency
    num_epochs_cnn = 3  # Reduced epochs with early stopping
    lr_cnn = 1e-4
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        
        self.image_paths = []
        self.labels = []
        for img_name in os.listdir(self.image_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.image_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(self.label_dir, f"{base_name}.txt")
                
                if not os.path.exists(label_path):
                    continue  # Skip images without labels
                
                has_fire = 0
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip() and line.split()[0] == '1':
                            has_fire = 1
                            break
                
                self.image_paths.append(img_path)
                self.labels.append(has_fire)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {str(e)}")
            return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return len(self.image_paths)

# CNN Feature Extractor (Optimized)
class FireFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all layers except the final classifier
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 2)  # Fine-tuning last layer only

    def forward(self, x):
        return self.cnn(x)

    def extract_features(self, x):
        # Extract features before the classification layer
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)
        x = self.cnn.avgpool(x)
        return torch.flatten(x, 1)

# Hybrid CNN-HDC Model
class HybridFireDetector:
    def __init__(self, config):
        self.config = config
        self.cnn_model = None
        self.hdc_encoder = None
        self.hdc_classifier = None
        
    def train_cnn(self):
        # Data augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = FireDataset(self.config.train_root, transform=transform)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.config.batch_size,
                                  shuffle=True,
                                  num_workers=4,  # Optimized
                                  pin_memory=True)

        # Initialize model and optimizer
        self.cnn_model = FireFeatureExtractor().to(self.config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn_model.cnn.fc.parameters(), lr=self.config.lr_cnn)


        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        print("\n--- Optimized CNN Training ---")
        for epoch in range(self.config.num_epochs_cnn):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():  # Use mixed precision
                    outputs = self.cnn_model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs_cnn}] Loss: {avg_loss:.4f}")

    def evaluate(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_dataset = FireDataset(self.config.test_root, transform=transform)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=4)

        y_true, y_pred = [], []
        inference_times = []

        print("\n--- Evaluation ---")
        self.cnn_model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Processing test data"):
                images = images.to(self.config.device)
                start_time = time.time()
                
                # Feature extraction
                outputs = self.cnn_model(images)
                batch_preds = torch.argmax(outputs, dim=1)

                inference_times.append(time.time() - start_time)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(batch_preds.cpu().numpy())

        # Compute evaluation metrics
        total_time = sum(inference_times)
        fps = len(test_dataset) / total_time
        print("\nEvaluation Results:")
        print(f"- Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
        print(f"- Precision:   {precision_score(y_true, y_pred):.4f}")
        print(f"- Recall:      {recall_score(y_true, y_pred):.4f}")
        print(f"- F1 Score:    {f1_score(y_true, y_pred):.4f}")
        print(f"- Inference Speed: {fps:.2f} FPS")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    config = Config()
    detector = HybridFireDetector(config)
    
    detector.train_cnn()  # Fast CNN training
    detector.evaluate()   # Run evaluation
