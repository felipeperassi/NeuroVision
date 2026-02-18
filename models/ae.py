import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Dataset 
class Roi2LatDataset(Dataset):
    def __init__(self, roi_data, target_data):
        self.roi = torch.tensor(roi_data, dtype=torch.float32)
        self.target = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.roi)

    def __getitem__(self, idx):
        return self.roi[idx], self.target[idx]

# AE Model
class Roi2LatAutoencoder(nn.Module):
    def __init__(self, input_dim=4657, latent_dim=1024):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), # input_dim -> 2048
            nn.BatchNorm1d(2048),       # Mean=0 & var=1
            nn.GELU(),
            nn.Dropout(0.5),            # High dropout

            nn.Linear(2048, latent_dim), # 2048 -> 1024
            nn.BatchNorm1d(latent_dim),
            nn.GELU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
