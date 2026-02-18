import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Dataset
class Roi2ClipDataset(Dataset):
    def __init__(self, voxel_data, clip_data):
        self.voxels = torch.tensor(voxel_data, dtype=torch.float32)
        raw_clip = torch.tensor(clip_data, dtype=torch.float32)
        self.clip = F.normalize(raw_clip, p=2, dim=-1)

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        return self.voxels[idx], self.clip[idx]

# MLP Model
class Roi2ClipMLP(nn.Module):
    def __init__(self, input_dim=4657, output_dim=768):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        raw_output = self.model(x)
        return F.normalize(raw_output, p=2, dim=-1) # L2 Norm for Clip space compatibility