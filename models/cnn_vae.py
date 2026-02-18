import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Dataset 
class Roi2VaeDataset(Dataset):
    def __init__(self, voxel_data, vae_data):
        self.voxels = torch.tensor(voxel_data, dtype=torch.float32)

        if len(vae_data.shape) == 2:
            self.vae = torch.tensor(vae_data, dtype=torch.float32).view(-1, 4, 64, 64)
        else:
            self.vae = torch.tensor(vae_data, dtype=torch.float32)

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        return self.voxels[idx], self.vae[idx]

# Residual Block: Memory-efficient deep CNNs
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return F.gelu(x + self.block(x))

# CNN Model
class Roi2VaeCNN(nn.Module):
    def __init__(self, voxel_dim=4657):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(voxel_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            # 4x4->8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256), 

            # 8x8->16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),

            # 16x16->32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),

            # 32x32->64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # 4 channels (64x64)
            nn.Conv2d(64, 4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4) # Trials -> 512x4x4
        x = self.decoder(x) # Trials -> 4x64x64 (Stable Diffusion VAE format)
        return x 