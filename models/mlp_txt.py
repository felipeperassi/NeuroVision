import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Dataset
class Clip2TxtDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input = torch.tensor(input_data, dtype=torch.float32)
        raw_output = torch.tensor(output_data, dtype=torch.float32)
        self.output = F.normalize(raw_output, p=2, dim=-1)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

# MLP Model
class Clip2TxtMLP(nn.Module):
    def __init__(self, input_dim=768, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(4096, seq_len * embed_dim)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.seq_len, self.embed_dim) # trials -> 77x768
        return F.normalize(x, p=2, dim=-1)
