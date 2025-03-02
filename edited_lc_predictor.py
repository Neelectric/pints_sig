import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data
tif_2017 = "2017_LU_data/gb2017lcm20m.tif"
tif_2023 = "2023_LU_data/gblcm2023_10m.tif"

# Random seed for reproducibility
np.random.seed(1234)

# Shuffle indices
indices = np.random.permutation(9100)
train_indices = indices[:8]
test_indices = indices[8090:]

# Image size for training
size = 500

# Convert index to coordinates
def index_to_coord(i):
    return i // 70, i % 70

# Dataset class for handling data
class LandUseDataset(Dataset):
    def __init__(self, indices, tif_2017, tif_2023):
        self.indices = indices
        self.tif_2017 = tif_2017
        self.tif_2023 = tif_2023

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        x, y = index_to_coord(index)

        # Load 2017 image
        with rasterio.open(self.tif_2017) as dataset:
            window = rasterio.windows.Window(x, y, 500, 500)
            img = dataset.read(1, window=window)  # Read band 1
            img[img == 255] = 0  # Replace no-data values

        # Load 2023 image (label)
        with rasterio.open(self.tif_2023) as dataset:
            window = rasterio.windows.Window(2 * x, 2 * y, 1000, 1000)
            label = dataset.read(1, window=window)
            label[label == 255] = 0  # Replace no-data values

        # Convert to PyTorch tensors
        img = torch.tensor(img, dtype=torch.long)  # No one-hot encoding
        label = torch.tensor(label, dtype=torch.long)  # No one-hot encoding

        return img, label

# Define DataLoader
batch_size = 8
train_dataset = LandUseDataset(train_indices, tif_2017, tif_2023)
test_dataset = LandUseDataset(test_indices, tif_2017, tif_2023)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim=1024):
        super(Autoencoder, self).__init__()

        # Encoder: Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # (64, 250, 250)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 125, 125)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 63, 63)
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Linear(256 * 63 * 63, bottleneck_dim)

        # Decoder: Upsampling
        self.decoder_fc = nn.Linear(bottleneck_dim, 256 * 32 * 32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 22, kernel_size=3, stride=2, padding=1, output_padding=1)  # (1, 512, 512)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Keep batch dimension
        x = self.bottleneck(x)
        x = self.decoder_fc(x)
        x = x.view(-1, 256, 32, 32)  # Reshape to decoder input size
        x = self.decoder(x)
        return x  # Output logits

# Initialize model
model = Autoencoder(bottleneck_dim=16).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use class indices, not one-hot
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_autoencoder(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
#            images, labels = images.to(device).unsqueeze(1), labels.to(device)  # Add channel dim
            images, labels = images.to(device).unsqueeze(1).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)

            # Resize output to match labels (512x512)
            outputs = F.interpolate(outputs, size=(1000, 1000), mode='bilinear', align_corners=False)

            loss = criterion(outputs, labels)  # CrossEntropyLoss needs shape [B, H, W]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation function
def eval_autoencoder(model, criterion, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device).unsqueeze(1).float(), labels.to(device).long()

            outputs = model(images)
            outputs = F.interpolate(outputs, size=(1000, 1000), mode='bilinear', align_corners=False)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

# Train and evaluate
print("Starting Training...")
train_autoencoder(model, criterion, optimizer, train_loader, epochs=10)
eval_autoencoder(model, criterion, test_loader)
