import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchhd
from torch.utils.data import Dataset, DataLoader
from torchhd import embeddings, functional, models as hd_models
from tqdm import tqdm
import torchmetrics
from PIL import Image
import numpy as np
import csv

# Configuration
class Config:
    img_size = 224  # Input image size
    channels = 3     # RGB channels
    num_levels = 1000
    dimensions = 10000
    batch_size = 1    # Reduce if memory constrained
    num_classes = 2   # Fire vs No Fire

    data_root = "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

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


# HDC Encoder (Adapted from MNIST example)
class FireEncoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super().__init__()
        self.flatten = nn.Flatten()
        self.position = embeddings.Random(
            size * size * 3,  # 224x224x3 RGB
            out_features
        )
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)  # Flatten to [batch, 224*224*3]
        x = x * (self.value.num_embeddings - 1)  # Scale to [0, num_levels-1]
        x = x.long()  # Convert to integer indices
        
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return functional.hard_quantize(sample_hv)

# Training Pipeline
def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_ds = FireDataset(f"{Config.data_root}/train", transform=transform)
    test_ds = FireDataset(f"{Config.data_root}/test", transform=transform)

    train_ld = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)

    # Initialize model components
    encoder = FireEncoder(Config.dimensions, Config.img_size, Config.num_levels)
    encoder = encoder.to(device)
    
    model = hd_models.Centroid(Config.dimensions, Config.num_classes)
    model = model.to(device)

    # Training phase
    train_samples = []
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            # Encode samples to hypervectors
            samples_hv = encoder(samples)
            model.add(samples_hv, labels)

            # Save training samples for analysis
            train_samples.append(samples_hv.cpu())

    # Save model parameters
    with open("fire_model_params.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for param_tensor in model.state_dict():
            param_data = model.state_dict()[param_tensor].cpu().numpy()
            writer.writerow([param_tensor])
            np.savetxt(csvfile, param_data.reshape(-1, 100), delimiter=",")
            writer.writerow([])

    # Save training samples
    train_samples = torch.cat(train_samples).numpy()
    np.savetxt("train_hvs.csv", train_samples, delimiter=",", fmt="%d")

    # Evaluation
    model.normalize()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    
    test_samples = []
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            
            samples_hv = encoder(samples)
            outputs = model(samples_hv, dot=True)
            
            accuracy.update(outputs, labels)
            test_samples.append(samples_hv.cpu())

    # Save test samples
    test_samples = torch.cat(test_samples).numpy()
    np.savetxt("test_hvs.csv", test_samples, delimiter=",", fmt="%d")

    print(f"\nTest Accuracy: {accuracy.compute().item()*100:.2f}%")

if __name__ == "__main__":
    main()