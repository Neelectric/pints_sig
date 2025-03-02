import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt

# --- Custom Dataset for Super-Resolution ---
class SuperResolutionDataset(Dataset):
    def __init__(self, low_res_images, high_res_lidar, transform=None):
        self.low_res_images = low_res_images  # List of low-res image tensors
        self.high_res_lidar = high_res_lidar
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        lr = self.low_res_images[idx]
        hrlid = self.high_res_lidar[idx]

        if self.transform:
            lr = self.transform(lr)
            hrlid = self.transform(hrlid)

        return lr, hrlid



# --- UNet-based Super-Resolution Model ---
class UNetSR(nn.Module):
    """
    UNet-based model for upscaling a 500x500x3 image to 5000x5000x3.
    Contains two downscaling layers and three upscaling layers.
    """

    def __init__(self):
        super(UNetSR, self).__init__()

        # Encoder (Downscaling)
        self.enc1 = self.conv_block(4, 64,2)
        self.enc2 = self.conv_block(64, 128,2)
        self.enc1y = self.conv_block(1, 4, 10)
        self.enc2y = self.conv_block(4, 64, 4)

        # Bottleneck
        self.bottleneck = self.bottlenecker(128, 256)
        self.bottlenecky = self.bottlenecker(64, 64)

        # Decoder (Upscaling)
        self.dec1 = self.upconv_block(256+64, 64,2)
        self.dec2 = self.upconv_block(64+64, 16,2)
        self.dec3 = self.upconv_block(20, 4,10)

    def conv_block(self, in_channels, out_channels,factor):
        """Convolutional block with Conv layers followed by ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=factor//2),
            nn.ReLU(),
            nn.MaxPool2d(factor),
            nn.BatchNorm2d(out_channels)# Downsampling
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
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x,y):
        """Forward pass through the UNet."""
        #First we downconv our low res satellite data
        e1 = self.enc1(x)  # 500 -> 250
        e2 = self.enc2(e1)  # 250 -> 125
        b = self.bottleneck(e2)  # 125 -> 125 (No downsampling here)

        #Second we downconv our high res lidar data
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

#SOme helper functions

def downscale(x, scale_factor=10):
    """Downsamples an image to low resolution."""
    return F.interpolate(x, scale_factor=1 / scale_factor, mode="bilinear", align_corners=True)

def self_supervised_loss(pred_high_res, input_low_res):
    """Encourages the downscaled output to match the input."""
    pred_low_res = downscale(pred_high_res)

    return F.mse_loss(pred_low_res + 1e-8, input_low_res + 1e-8)

def total_variation_loss(img):
    """Encourages smooth textures in output image."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def lidar_structure_loss(pred_rgb, lidar):
    """Encourages structure preservation from LiDAR."""
    return F.l1_loss(pred_rgb, lidar.repeat(1, 4, 1, 1))  # Convert LiDAR to 3-channel

# --- Training Function ---
def train_model(model, dataloader, epochs=10, lr=1):
    """
    Train the UNet model for super-resolution.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)


    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for lr_imgs, hr_lidar in dataloader:

            output = model(lr_imgs,hr_lidar)  # Forward pass
            loss1 = self_supervised_loss(output, lr_imgs)
            loss2 = total_variation_loss(output)
            loss3 = 0;lidar_structure_loss(output, hr_lidar)
            loss = loss1 + 0.01 * loss2 + 0.1 * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    return model


# --- Initialize Model ---
if __name__ == "__main__":
    # Example dataset (replace with real images)

    #low_res_images = [torch.ones(4, 500, 500) for _ in range(1)]
    #high_res_lidar = [torch.randn(1, 5000, 5000) for _ in range(1)]

    with rasterio.open(r"C:\Users\mgbsa\Downloads\HackathonData\HackathonData\20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif") as rbg:
        tiff_rbg = rbg.read()
        tiff_rbg = tiff_rbg/np.max(tiff_rbg)# Reads as a NumPy array
        low_res_images = [torch.tensor(tiff_rbg).float()]

    with rasterio.open(r"C:\Users\mgbsa\Downloads\HackathonData\HackathonData\DSM_TQ0075_P_12757_20230109_20230315.tif") as lid:
        tiff_lidar = lid.read()
        tiff_lidar = tiff_lidar / np.max(tiff_lidar)
        high_res_lidar = [torch.tensor(tiff_lidar).float()]

    dataset = SuperResolutionDataset(low_res_images, high_res_lidar, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNetSR()
    trained_model = train_model(model, dataloader, epochs=10)
    for lr_imgs, hr_lidar in dataloader:
        new = trained_model.forward(lr_imgs, hr_lidar).detach().numpy().reshape(4,5000,5000)
        new = np.transpose(new, (1, 2, 0))
        plt.imshow(new)
        plt.show()