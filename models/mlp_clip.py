import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelToCLIP(nn.Module):
    """
    MLP Model to predict CLIP features from voxel data.
    
    Input: (trials, 15724) voxel vector.
    
    Output: CLIP feature vector, 1024 dimensions.
    
    MLP Architecture: Voxels (15724) -> Hidden (4096 -> 2048 -> 1024) -> CLIP (1024)
    """
    def __init__(self, input_dim: int, clip_dim: int = 1024) -> None:
        """
        Initializes the VoxelToCLIP MLP model.
            Args:
                input_dim (int): The dimensionality of the input voxel vector (e.g., 15724).
                clip_dim (int): The dimensionality of the output CLIP feature vector (default: 1024).
        """
        super().__init__()

        # MLP: Voxels (15724) -> Hidden (4096 -> 2048 -> 1024) -> CLIP (1024)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.GELU(),
            nn.LayerNorm(4096),
            nn.Dropout(0.5),    

            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, clip_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
            Args:
                x (torch.Tensor): Input voxel data (trials, 15724).
            Returns:
                torch.Tensor: Output normalized CLIP feature vector (trials, 1024).
        """
        return F.normalize(self.mlp(x), dim=-1)