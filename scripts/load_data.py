from config import IDX_TEST, IDX_TRAIN, DATA_VOXELS_ST, DEVICE
from models import VoxelToCLIP, VoxelToVGG, VoxelToVAE

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, AutoencoderKL, ControlNetModel

class VoxelTargetDataset(Dataset):
    def __init__(self, voxels : torch.Tensor, t1 : torch.Tensor, t2 : torch.Tensor = None, t3 : torch.Tensor = None, 
                 target_name : str = None, indices : np.ndarray = None, 
                 mean : torch.Tensor = None, std : torch.Tensor = None) -> None:
        """
        Initializes the VoxelTargetDataset.
            Args:
                - voxels (torch.Tensor): Voxel data tensor of shape (trials, input_dim).
                - t1 (torch.Tensor): Target tensor 1 (e.g., CLIP, VAE, or VGG features).
                - t2 (torch.Tensor, optional): Target tensor 2 (e.g., VGG features). Required if target_name is "VGG".
                - t3 (torch.Tensor, optional): Target tensor 3 (e.g., VGG features). Required if target_name is "VGG".
                - target_name (str): Name of the target type ("CLIP", "VAE", or "VGG").
                - indices (np.ndarray): Array of trial indices to include in the dataset.
                - mean (torch.Tensor): Mean voxel values for normalization.
                - std (torch.Tensor): Standard deviation of voxel values for normalization.
        """
        self.name = target_name.upper() if target_name else None
        names = ["CLIP", "VAE", "VGG"]
        if self.name not in names:
            raise ValueError(f"Invalid target name: {target_name}. Must be one of {names}.")
        
        self.voxels = voxels
        if self.name == "CLIP" or self.name == "VAE":
            self.t1 = t1
        else:
            self.t1     = t1
            self.t2     = t2
            self.t3     = t3 

        self.indices = indices
        self.mean    = mean
        self.std     = std

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, i : int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the i-th sample from the dataset, applying normalization to the voxel data.
            Args:
                - i (int): Index of the sample to retrieve.
            Returns:
                - If target_name is "CLIP" or "VAE": (normalized_voxel, t1)
                - If target_name is "VGG": (normalized_voxel, t1, t2, t3)
        """
        idx = self.indices[i]
        x = (self.voxels[idx].squeeze() - self.mean.squeeze()) / self.std.squeeze()
        if self.name == "CLIP" or self.name == "VAE": return x, self.t1[idx]
        else: return x, self.t1[idx], self.t2[idx], self.t3[idx]


def load_training_data(voxels_path : str, t1_path : str, t2_path : str = None, t3_path : str = None, 
              target_name : str = None, batch_size : int = None) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Loads the voxel and target data, splits it into training and testing sets, and returns DataLoaders for each set.
        Args:
            - voxels_path (str): Path to the voxel data .npy file.
            - t1_path (str): Path to the target tensor 1 .npy file (e.g., CLIP, VAE, or VGG features).
            - t2_path (str, optional): Path to the target tensor 2 .npy file (e.g., VGG features). Required if target_name is "VGG".
            - t3_path (str, optional): Path to the target tensor 3 .npy file (e.g., VGG features). Required if target_name is "VGG".
            - target_name (str): Name of the target type ("CLIP", "VAE", or "VGG").
            - batch_size (int): Batch size for the DataLoaders.
        Returns:
            - tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Training DataLoader, Testing DataLoader, Mean tensor, Std tensor.
    """
    target_name = target_name.upper() if target_name else None
    if target_name not in ["CLIP", "VAE", "VGG"]:
        raise ValueError(f"Invalid target name: {target_name}. Must be one of ['CLIP', 'VAE', 'VGG'].")
    
    voxels = torch.tensor(np.load(voxels_path), dtype=torch.float32)
    if target_name == "CLIP" or target_name == "VAE":
        t1 = torch.tensor(np.load(t1_path), dtype=torch.float32)
        t2 = t3 = None
    else:
        t1  = torch.tensor(np.load(t1_path),   dtype=torch.float32)
        t2  = torch.tensor(np.load(t2_path),   dtype=torch.float32) 
        t3  = torch.tensor(np.load(t3_path),   dtype=torch.float32) 

    train_indices = np.load(IDX_TRAIN)
    test_indices  = np.load(IDX_TEST)

    stats = np.load(DATA_VOXELS_ST)
    mean = torch.tensor(stats['mean'], dtype=torch.float32)
    std  = torch.tensor(stats['std'],  dtype=torch.float32)

    train_loader = DataLoader(
        VoxelTargetDataset(voxels, t1, t2, t3, target_name, train_indices, mean, std),
        batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        VoxelTargetDataset(voxels, t1, t2, t3, target_name, test_indices, mean, std),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, mean, std

def load_inference_pipeline(cnn_name : str) -> tuple[StableDiffusionImg2ImgPipeline | StableDiffusionControlNetPipeline, AutoencoderKL | ControlNetModel]:
    """
    Loads the Stable Diffusion pipeline and corresponding model (VAE or ControlNet) for inference based on the specified CNN name.
        Args:
            - cnn_name (str): Name of the CNN model ("VAE" or "VGG") to determine which pipeline and model to load.
        Returns:
            - tuple[StableDiffusionImg2ImgPipeline | StableDiffusionControlNetPipeline, AutoencoderKL | ControlNetModel]: The loaded Stable Diffusion pipeline and the corresponding model (VAE or ControlNet).
    """
    names = ["VAE", "VGG"]
    if cnn_name.upper() not in names:
        raise ValueError(f"Invalid CNN name: {cnn_name}. Must be one of {names}.")

    if cnn_name.upper() == "VAE":
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(DEVICE).eval()

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(DEVICE)
        pipe.safety_checker = None
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(1.0)

        return pipe, vae
    else:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        ).to(DEVICE)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(DEVICE)
        pipe.safety_checker = None
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(1.0)

        return pipe, controlnet


def load_inference_data(voxels_path : str, mlp_path : str, cnn_path : str, cnn_name : str, 
                            voxels_st_path : str, idx_test_path : str, 
                            idx_image : int) -> tuple[StableDiffusionImg2ImgPipeline | StableDiffusionControlNetPipeline, AutoencoderKL | ControlNetModel, torch.Tensor, VoxelToCLIP, VoxelToVAE | VoxelToVGG]:
    """
    Loads the necessary data and models for performing inference on a single trial based on the specified CNN name.
        Args:
            - voxels_path (str): Path to the voxel data .npy file.
            - mlp_path (str): Path to the saved MLP model weights .pt file.
            - cnn_path (str): Path to the saved CNN model weights .pt file.
            - cnn_name (str): Name of the CNN model ("VAE" or "VGG") to determine which pipeline and model to load.
            - voxels_st_path (str): Path to the voxel statistics .npz file.
            - idx_test_path (str): Path to the test indices .npy file.
            - idx_image (int): Index of the image in the test set to process.
        Returns:
            - tuple[StableDiffusionImg2ImgPipeline | StableDiffusionControlNetPipeline, AutoencoderKL | ControlNetModel, 
            torch.Tensor, VoxelToCLIP, VoxelToVAE | VoxelToVGG]: The loaded Stable Diffusion pipeline, corresponding model (VAE or ControlNet), 
            normalized voxel tensor for the specified trial, loaded MLP model, and loaded CNN model.
        """
    names = ["VAE", "VGG"]
    if cnn_name.upper() not in names:
        raise ValueError(f"Invalid CNN name: {cnn_name}. Must be one of {names}.")
    
    # Load pipeline and models for inference
    pipe, model = load_inference_pipeline(cnn_name)

    # Process voxel data
    voxels = torch.tensor(np.load(voxels_path), dtype=torch.float32).to(DEVICE)
    voxels_st = np.load(voxels_st_path)
    mean, std = torch.tensor(voxels_st['mean'], dtype=torch.float32).to(DEVICE), torch.tensor(voxels_st['std'], dtype=torch.float32).to(DEVICE)
    voxels_norm = (voxels - mean) / std

    # Idx for inference
    idx_test = np.load(idx_test_path)
    idx = idx_test[idx_image]
    voxel_idx = voxels_norm[idx].unsqueeze(0) 
   
   # Load MLP
    mlp_weights = torch.load(mlp_path, map_location=DEVICE)
    mlp = VoxelToCLIP(input_dim=mlp_weights['input_dim']).to(DEVICE)
    mlp.load_state_dict(mlp_weights['model_state'])
    mlp.eval()

    # Load CNN
    cnn_weights = torch.load(cnn_path, map_location=DEVICE)
    if cnn_name.upper() == "VAE":
        cnn = VoxelToVAE(input_dim=cnn_weights['input_dim']).to(DEVICE)
    else:
        cnn = VoxelToVGG(input_dim=cnn_weights['input_dim']).to(DEVICE)
    cnn.load_state_dict(cnn_weights['model_state'])
    cnn.eval()
    
    return pipe, model, voxel_idx, mlp, cnn
