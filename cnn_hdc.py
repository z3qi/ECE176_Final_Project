"""
Wildfire Detection System - CNN, HDC, and Hybrid Models
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

import torchhd
from torchhd import embeddings, functional, models as hd_models
from torchhd.models import Centroid as Centroid

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
    cnn_feature_dim = 512  # ResNet18 feature dimension 512
    hdc_dim = 4096         # Hypervector dimension 4096
    quantize = "binary"    # "binary" or "ternary"
    
    # Training parameters
    batch_size = 16         # 16
    num_epochs_cnn = 2     # CNN training epochs 3
    lr_cnn = 1e-4
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Custom Dataset Class
class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        for img_name in os.listdir(self.image_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.image_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(self.label_dir, f"{base_name}.txt")
                
                # Label parsing
                has_fire = 0
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0] == '1':
                            has_fire = 1
                            break
                
                self.image_paths.append(img_path)
                self.labels.append(has_fire)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 1. CNN-Only Model
class CNNModel:
    def __init__(self, config):
        self.config = config
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.to(config.device)
        
    def train(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_cnn)
        
        print("\n--- Training CNN-only Model ---")
        for epoch in range(self.config.num_epochs_cnn):
            self.model.train()
            running_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f"Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader):
        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.config.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        return self._calculate_metrics(y_true, y_pred)
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

# 2. HDC-Only Model
class HDCModel:
    def __init__(self, config):
        self.config = config
        # self.encoder = embeddings.Projection(
        #     input_size=224*224*3,  # Raw pixels
        #     dimensions=config.hdc_dim
        # ).to(config.device)
        self.encoder = embeddings.Projection(
            in_features=224*224*3,  # Changed from input_size
            out_features=config.hdc_dim  # Changed from dimensions
        ).to(config.device)
        
        self.classifier = Centroid(
            out_features=2,
            in_features=config.hdc_dim
        ).to(config.device)

        # self.classifier = Centroid(2, config.hdc_dim).to(config.device)
        
    def train(self, train_loader):
        print("\n--- Training HDC-only Model ---")
        for images, labels in tqdm(train_loader, desc="Processing"):
            images = images.view(-1, 224*224*3).to(self.config.device)  # Flatten
            hypervectors = self.encoder(images)
            hypervectors = functional.hard_quantize(hypervectors)
            self.classifier.add(hypervectors, labels.to(self.config.device))
        
        self.classifier.normalize()

    def evaluate(self, test_loader):
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.view(-1, 224*224*3).to(self.config.device)
                hypervectors = self.encoder(images)
                hypervectors = functional.hard_quantize(hypervectors)
                logits = self.classifier(hypervectors, dot=False)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        return self._calculate_metrics(y_true, y_pred)
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

# 3. Hybrid CNN-HDC Model
class HybridModel:
    def __init__(self, config):
        self.config = config
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # Remove final classification layer
        self.cnn.to(config.device)
        
        # self.hdc_encoder = embeddings.Projection(
        #     input_size=config.cnn_feature_dim,
        #     dimensions=config.hdc_dim
        # ).to(config.device)
        self.hdc_encoder = embeddings.Projection(
            in_features=config.cnn_feature_dim,
            out_features=config.hdc_dim
        ).to(config.device)
        
        self.classifier = Centroid(
            out_features=2,
            in_features=config.hdc_dim,
        ).to(config.device)

        # self.classifier = Centroid(2, config.hdc_dim).to(config.device)

    def train_cnn(self, train_loader):
        # Temporarily add classification head for CNN training
        original_fc = self.cnn.fc
        self.cnn.fc = nn.Linear(self.config.cnn_feature_dim, 2).to(self.config.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.config.lr_cnn)
        
        print("\n--- Training CNN Feature Extractor ---")
        for epoch in range(self.config.num_epochs_cnn):
            self.cnn.train()
            running_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.cnn(images)  # Class scores
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f"Loss: {avg_loss:.4f}")
        
        # Restore feature extractor mode
        self.cnn.fc = original_fc

    def train_hdc(self, train_loader):
        print("\n--- Training HDC Classifier ---")
        self.cnn.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Encoding features"):
                images = images.to(self.config.device)
                features = self.cnn(images)  # Get 512D features
                hypervectors = self.hdc_encoder(features)
                
                if self.config.quantize == "binary":
                    hypervectors = functional.hard_quantize(hypervectors)
                elif self.config.quantize == "ternary":
                    hypervectors = functional.ternary_quantize(hypervectors)
                
                self.classifier.add(hypervectors, labels.to(self.config.device))
        
        self.classifier.normalize()

    def evaluate(self, test_loader):
        y_true, y_pred = [], []
        self.cnn.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.config.device)
                features = self.cnn(images)
                hypervectors = self.hdc_encoder(features)
                
                if self.config.quantize == "binary":
                    hypervectors = functional.hard_quantize(hypervectors)
                elif self.config.quantize == "ternary":
                    hypervectors = functional.ternary_quantize(hypervectors)
                
                logits = self.classifier(hypervectors, dot=False)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

# Main Workflow
if __name__ == "__main__":
    config = Config()
    
    # Common transforms
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    hdc_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load datasets
    train_dataset_cnn = FireDataset(config.train_root, cnn_transform)
    train_dataset_hdc = FireDataset(config.train_root, hdc_transform)
    test_dataset = FireDataset(config.test_root, cnn_transform)
    
    # Create models
    cnn_model = CNNModel(config)
    hdc_model = HDCModel(config)
    hybrid_model = HybridModel(config)
    
    # Train and evaluate
    results = {}
    
    # CNN-only
    cnn_loader = DataLoader(train_dataset_cnn, batch_size=config.batch_size, shuffle=True)
    cnn_model.train(cnn_loader)
    results["CNN-only"] = cnn_model.evaluate(DataLoader(test_dataset, batch_size=config.batch_size))
    
    # HDC-only
    hdc_loader = DataLoader(train_dataset_hdc, batch_size=config.batch_size, shuffle=True)
    hdc_model.train(hdc_loader)
    results["HDC-only"] = hdc_model.evaluate(DataLoader(test_dataset, batch_size=config.batch_size))
    
    # Hybrid
    hybrid_model.train_cnn(cnn_loader)
    hybrid_model.train_hdc(cnn_loader)
    results["Hybrid"] = hybrid_model.evaluate(DataLoader(test_dataset, batch_size=config.batch_size))
    
    # Print results
    print("\nFinal Results:")
    for model, metrics in results.items():
        print(f"\n--- {model} ---")
        for k, v in metrics.items():
            if k == "Confusion Matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v:.4f}")