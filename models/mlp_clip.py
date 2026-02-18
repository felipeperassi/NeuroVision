import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Dataset
class Voxels2ClipDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input = torch.tensor(input_data, dtype=torch.float32)
        raw_output = torch.tensor(output_data, dtype=torch.float32)
        self.output = F.normalize(raw_output, p=2, dim=-1)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

# MLP Model
class Voxels2ClipMLP(nn.Module):
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