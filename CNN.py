import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- Custom Dataset for Super-Resolution ---
class SuperResolutionDataset(Dataset):
    def __init__(self, low_res_images, high_res_images,high_res_lidar, transform=None):
        self.low_res_images = low_res_images  # List of low-res image tensors
        self.high_res_images = high_res_images  # List of high-res image tensors
        self.high_res_lidar = high_res_lidar
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        lr = self.low_res_images[idx]
        hr = self.high_res_images[idx]
        hrlid = self.high_res_lidar[idx]

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
            hrlid = self.transform(hrlid)

        return lr, hr, hrlid



# --- UNet-based Super-Resolution Model ---
class UNetSR(nn.Module):
    """
    UNet-based model for upscaling a 500x500x3 image to 5000x5000x3.
    Contains two downscaling layers and three upscaling layers.
    """

    def __init__(self):
        super(UNetSR, self).__init__()

        # Encoder (Downscaling)
        self.enc1 = self.conv_block(3, 64,2)
        self.enc2 = self.conv_block(64, 128,2)
        self.enc1y = self.conv_block(1, 3, 10)
        self.enc2y = self.conv_block(3, 64, 4)

        # Bottleneck
        self.bottleneck = self.bottlenecker(128, 256)
        self.bottlenecky = self.bottlenecker(64, 64)

        # Decoder (Upscaling)
        self.dec1 = self.upconv_block(256+64, 64,2)
        self.dec2 = self.upconv_block(64+64, 16,2)
        self.dec3 = self.upconv_block(19, 3,10)

    def conv_block(self, in_channels, out_channels,factor):
        """Convolutional block with Conv layers followed by ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=factor//2),
            nn.ReLU(),
            nn.MaxPool2d(factor)  # Downsampling
        )

    def bottlenecker(self, in_channels, out_channels):
        """Convolutional block with Conv layers followed by ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())

    def upconv_block(self, in_channels, out_channels, scale_factor=2):
        """Upsampling block with transposed convolution allowing different scale factors."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor,
                               padding=scale_factor // 2),
            nn.ReLU()
        )

    def forward(self, x,y):
        """Forward pass through the UNet."""
        #First we downconv our low res satellite data
        e1 = self.enc1(x)  # 500 -> 250
        e2 = self.enc2(e1)  # 250 -> 125
        b = self.bottleneck(e2)  # 125 -> 125 (No downsampling here)

        #Second we downconv our high res satellite data
        ey1 = self.enc1y(y)
        e2y = self.enc2y(ey1)
        by = self.bottlenecky(e2y)

        #Then we concatenate the two
        b = torch.cat((b,by),dim=1)


        d1 = self.dec1(b)  # 125 -> 250
        d1p = torch.cat((d1, e1), dim=1)
        d2 = self.dec2(d1p)  # 250 -> 500
        d2p = torch.cat((d2, x), dim=1)
        out = self.dec3(d2p)  # 500 -> 5000

        return torch.sigmoid(out)  # Output in range [0,1]


# --- Training Function ---
def train_model(model, dataloader, epochs=10, lr=1e-4):
    """
    Train the UNet model for super-resolution.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # L1 loss for sharper textures
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for lr_imgs, hr_imgs, hr_lidar in dataloader:

            output = model(lr_imgs,hr_lidar)  # Forward pass
            loss = criterion(output, hr_imgs)  # Compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    return model


# --- Initialize Model ---
if __name__ == "__main__":
    # model = UNetSR()
    # test_input = torch.randn(1, 3, 500, 500)
    # test_input_hr = torch.randn(1,1,5000,5000)# Batch size 1, 3 RGB channels, 500x500 image
    # output = model(test_input,test_input_hr)
    # print("Output shape:", output.shape)  # Expected: (1, 3, 1000, 1000)

    # Example dataset (replace with real images)
    low_res_images = [torch.randn(3, 500, 500) for _ in range(5)]
    high_res_images = [torch.randn(3, 5000, 5000) for _ in range(5)]
    high_res_lidar = [torch.randn(1, 5000, 5000) for _ in range(5)]

    dataset = SuperResolutionDataset(low_res_images, high_res_images, high_res_lidar, transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNetSR()
    trained_model = train_model(model, dataloader, epochs=10)