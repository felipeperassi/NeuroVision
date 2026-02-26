import torch
import torch.nn as nn

class VoxelToVGG(nn.Module):
    """
    CNN Model to predict VGG features from voxel data.
    
    Input: (trials, 15724) voxel vector.

    Output: Tuple of 3 feature maps corresponding to VGG layers:
    - H1: Low level features, 64 channels (14x14).
    - H2: Low-Mid level features, 128 channels (14x14).
    - H3: Mid level features, 256 channels (14x14).
   
     Architecture:
    - FC: voxels -> 512 channels (2x2).
    - Upsampling 512 channels (2x2) -> 128 channels (8x8)
    - 3 heads, one per VGG layer:
        - 1-5 VGG layers: 64 channels (14x14).
        - 1-10 VGG layers: 128 channels (14x14).
        - 1-15 VGG layers: 256 channels (14x14).
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initializes the VoxelToVGG model.
            Args:
                input_dim (int): The dimensionality of the input voxel vector (e.g., 15724).
        """
        super().__init__()

        # Voxels 15724 -> 512 (2x2)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512 * 2 * 2)    # 512 channels (2x2)
        )

        # Upsampling 512 (2x2) -> 128 (8x8)
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256 channels (4x4)
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 256 channels (8x8)
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # 128 channels (8x8) 
            nn.GELU(),
        )

        # 3 heads, one per VGG layer
        # H1: Low level features, 64 (14x14)
        self.head1 = nn.Sequential( 
            nn.ConvTranspose2d(128, 64, 3, 1, 1),   # 64 channels (8x8) 
            nn.GELU(),
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),   # 64 channels (14x14) 
            nn.Conv2d(64, 64, 3, 1, 1) 
        )
        
        # H2: Low-Mid level features, 128 (14x14)
        self.head2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, 1),  # 128 channels (8x8) 
            nn.GELU(),  
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),   # 128 channels (14x14) 
            nn.Conv2d(128, 128, 3, 1, 1)
        )

        # H3: Mid level features, 256 (14x14)
        self.head3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 1, 1),  # 256 channels (8x8)
            nn.GELU(),
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),   # 256 channels (14x14) 
            nn.Conv2d(256, 256, 3, 1, 1)            
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
            Args:
                x (torch.Tensor): Input voxel data (trials, 15724).
            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                    - H1: 64 channels (14x14).
                    - H2: 128 channels (14x14).
                    - H3: 256 channels (14x14).
        """
        x = self.fc(x).view(-1, 512, 2, 2)
        x = self.backbone(x)
        return self.head1(x), self.head2(x), self.head3(x)