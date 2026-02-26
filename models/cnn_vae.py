import torch
import torch.nn as nn

class VoxelToVAE(nn.Module):
    """
    CNN Model to predict VAE features from voxel data.
    
    Input: (trials, 15724) voxel vector.

    Output: Structural features, 4 channels (64x64)

    Architecture:
    - FC: voxels -> 256 channels (8x8)
    - Upsampling 256 channels (8x8) -> 32 channels (64x64)
    - Head: Latent VAE, 4 channels (64x64)
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initializes the VoxelToVAE model.
            Args:
                input_dim (int): The dimensionality of the input voxel vector (e.g., 15724).
        """
        super().__init__()

        # Voxels 15724 -> 256 (8×8)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256 * 8 * 8)    # 256 channels (8x8)
        )

        # Upsampling 256 (8×8) -> 32 (64×64)
        self.backbone = nn.Sequential(
            self._upsampling_block(256, 128),  # 128 channels (16×16)
            self._upsampling_block(128, 64),   # 64 channels (32×32)
            self._upsampling_block(64,  32),   # 32 channels (64×64)
        )

        # Latent VAE: Structural features, 4 (64×64)
        self.head = nn.Conv2d(32, 4, 3, 1, 1)

    def _upsampling_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Creates an upsampling block consisting of:
        - Upsampling by a factor of 2 (bilinear).
        - Convolution to reduce channels.
        - Batch Normalization.
        - GELU activation.
            Args:
                in_ch (int): Number of input channels.
                out_ch (int): Number of output channels.
            Returns:
                nn.Sequential: The upsampling block.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
            Args:
                x (torch.Tensor): Input voxel data (trials, 15724).
            Returns:
                torch.Tensor: Output feature maps (trials, 4, 64, 64).
        """
        x = self.fc(x).view(-1, 256, 8, 8)
        x = self.backbone(x)
        return self.head(x)  # (B, 4, 64, 64)