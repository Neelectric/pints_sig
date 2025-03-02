import rasterio
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

# Path to your .tif file
tif_2017 = "2017_LU_data/gb2017lcm20m.tif"
tif_2023 = "2023_LU_data/gblcm2023_10m.tif"

def convert_2017_to_2023(x, y):
    return 2 * x, 2 * y

def index_to_coord(i):
    return i // 70, i % 70

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder: Downsampling with Conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(22, 64, kernel_size=3, stride=2, padding=1),  # (64, 250, 250)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 125, 125)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 63, 63)
            nn.ReLU(),
        )

        # Bottleneck fully connected layer
        self.bottleneck = nn.Linear(256 * 63 * 63, 1024)

        # Decoder: Upsampling with Transposed Conv layers
        self.decoder_fc = nn.Linear(1024, 256 * 125 * 125)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 250, 250)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 500, 500)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 1000, 1000)
            nn.ReLU(),
            nn.Conv2d(32, 22, kernel_size=3, padding=1)  # (21, 1000, 1000) - No activation (raw logits)
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.bottleneck(x)

        # Decode
        x = self.decoder_fc(x)
        x = x.view(x.shape[0], 256, 125, 125)
        x = self.decoder(x)
        
        return x  # Raw logits, ready for CrossEntropyLoss

class TestAutoencoder(nn.Module):
    def __init__(self):
        super(TestAutoencoder, self).__init__()

        # Encoder: Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(22, 64, kernel_size=3, stride=2, padding=1),  # (64, 250, 250)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 125, 125)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 63, 63)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # (256, 32, 32)
            nn.ReLU(),
        )

        # No Fully Connected Layer (Memory-efficient)
        # Decoder: Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),  # (256, 63, 63)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),  # (128, 125, 125)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 250, 250)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 500, 500)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 22, kernel_size=3, stride=2, padding=1, output_padding=1)  # (21, 1000, 1000)
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Raw logits, ready for CrossEntropyLoss


np.random.seed(1234)

indices = np.random.permutation(9100)

train_indices = indices[:8090]
test_indices = indices[8090:]

size = 500

def get_image(index):
    x, y = index_to_coord(index)
    with rasterio.open(tif_2017) as dataset:
        window = rasterio.windows.Window(x, y, 500, 500)
        first_band_subset = dataset.read(1, window=window)  # Read band 1
        first_band_subset[first_band_subset == 255] = 0
    first_band_subset = torch.from_numpy(first_band_subset.astype(np.int64))
    first_band_subset = torch.nn.functional.one_hot(first_band_subset, num_classes=22)
    return first_band_subset.permute(2, 0, 1).float()

def get_label(index):
    x, y = index_to_coord(index)
    with rasterio.open(tif_2023) as dataset:
        window = rasterio.windows.Window(2 * x, 2 * y, 1000, 1000)
        first_band_subset = dataset.read(1, window=window)  # Read band 1
        first_band_subset[first_band_subset == 255] = 0
    first_band_subset = torch.from_numpy(first_band_subset.astype(np.int64))
    first_band_subset = torch.nn.functional.one_hot(first_band_subset, num_classes=22)
    return first_band_subset.permute(2, 0, 1).float()

print("Loading Model")

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

print("Model loaded")

# Loss function: CrossEntropyLoss for pixel classification
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_autoencoder(model, criterion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for index in train_indices:
            image = get_image(index).to(device)
            label = get_label(index).to(device)

            optimizer.zero_grad()
            output = model(image)  # Shape: [batch, 21, 1000, 1000]

            # Compute loss (logits vs class labels)
            loss = criterion(output, label)  # Labels shape: [batch, 1000, 1000]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_indices):.4f}")

def eval_autoencoder(model, criterion):
    model.eval()

    total_loss = 0
    for index in test_indices:
        image = get_image(index).to(device)
        label = get_label(index).to(device)

        optimizer.zero_grad()
        output = model(image)  # Shape: [batch, 21, 1000, 1000]

        # Compute loss (logits vs class labels)
        loss = criterion(output, label)  # Labels shape: [batch, 1000, 1000]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_indices):.4f}")

print("Beginning Training")

train_autoencoder(model, criterion, optimizer, epochs=10)
eval_autoencoder(model, criterion)
