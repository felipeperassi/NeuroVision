from config import (
    DATA_VOXELS, DATA_VOXELS_ST, IDX_TEST, 
    WEIGHTS_CLIP, WEIGHTS_VAE, WEIGHTS_VGG, DIR_RESULTS
)
from load_data import load_inference_data

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, ControlNetModel
from models import VoxelToVGG, VoxelToVAE
import argparse

def process_inference(model: AutoencoderKL | ControlNetModel, cnn_name : str, voxel_idx : torch.Tensor, cnn : VoxelToVGG | VoxelToVAE) -> Image.Image:
    """
    Processes inference for a given model, CNN name, voxel index, and CNN model.
        Args:
            - model (AutoencoderKL | ControlNetModel): The loaded Stable Diffusion model (VAE or ControlNet) to be used for processing the inference.
            - cnn_name (str): Name of the CNN model ("VAE" or "VGG") to determine which processing pipeline to use.
            - voxel_idx (torch.Tensor): The input voxel data tensor for the specified image index, normalized and ready for inference.
            - cnn (VoxelToVGG | VoxelToVAE): The loaded CNN model corresponding to the specified CNN name, used to generate predictions from the voxel data.
        Returns:
            - Image.Image: The processed image generated from the Stable Diffusion pipeline based on the input voxel data and CNN model.
    """
    if cnn_name.upper() == "VAE":
        with torch.no_grad():
            vae_pred = cnn(voxel_idx)
            # vae_pred      = vae_pred_norm * std_vae.view(1, 4, 64, 64) \
            #             + mean_vae.view(1, 4, 64, 64)
            vae_pred      = vae_pred / 0.18215
            decoded       = model.decode(vae_pred).sample
            decoded       = (decoded.clamp(-1, 1) + 1) / 2
            init_img = Image.fromarray(
            (decoded.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).resize((512, 512))

            return init_img

    else:
        with torch.no_grad():
            f1_pred, _, _ = cnn(voxel_idx)
            # f1_denorm = f1_pred * stats1[1] + stats1[0]
            f1_avg    = f1_pred.mean(dim=1, keepdim=True)
            depth_map = F.interpolate(f1_avg, (512, 512), mode='bilinear', align_corners=False)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            # Convertir a RGB (ControlNet depth espera RGB)
            depth_np  = (depth_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
            depth_rgb = Image.fromarray(depth_np).convert('RGB')

            return depth_rgb


def inference(cnn_name : str, idx_image : int) -> Image.Image:
    """
    Performs inference using the specified CNN model and image index.
        Args:
            - cnn_name (str): Name of the CNN model ("VAE" or "VGG") to determine which pipeline and model to load.
            - idx_image (int): Index of the image to perform inference on.
        Returns:
            - Image.Image: The generated image from the Stable Diffusion pipeline based on the input voxel data and CNN model.
    """
    names = ["VAE", "VGG"]
    if cnn_name.upper() not in names:
        raise ValueError(f"Invalid CNN name: {cnn_name}. Must be one of {names}.")
    
    if cnn_name.upper() == "VAE": cnn_path = WEIGHTS_VAE
    else: cnn_path = WEIGHTS_VGG

    pipe, model, voxel_idx, mlp, cnn = load_inference_data(
                                    voxels_path=DATA_VOXELS, mlp_path=WEIGHTS_CLIP, voxels_st_path=DATA_VOXELS_ST, 
                                    idx_test_path=IDX_TEST, cnn_path=cnn_path, cnn_name=cnn_name, idx_image=idx_image
                                )

    with torch.no_grad():
        clip_pred    = mlp(voxel_idx).float().squeeze(0)
    neg_embeds   = torch.zeros_like(clip_pred).unsqueeze(0)
    pos_embeds   = clip_pred.unsqueeze(0)
    image_embeds = torch.cat([neg_embeds, pos_embeds], dim=0).unsqueeze(1).half()

    structured_img = process_inference(model, cnn_name, voxel_idx, cnn)

    if cnn_name.upper() == "VAE":
        output = pipe(
            prompt="",
            image=structured_img,
            strength=0.8,
            ip_adapter_image_embeds=[image_embeds],
            negative_prompt="blurry, distorted, low quality",
            num_inference_steps=50,
            guidance_scale=8.0
        ).images[0]

    else:
        output = pipe(
            prompt="",
            image=structured_img,
            ip_adapter_image_embeds=[image_embeds],
            negative_prompt="blurry, distorted, low quality",
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.5
        ).images[0]

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master inference script.')
    parser.add_argument('--cnn', type=str, required=True, 
                        choices=['VAE', 'VGG'], help='Select model: VAE or VGG.')
    parser.add_argument('--idx', type=int, required=True, 
                        help='Index of the image to be processed.', 
                        choices=range(3000))
    
    args = parser.parse_args()
    output = inference(cnn_name=args.cnn, idx_image=int(args.idx))
    output.save(f"{DIR_RESULTS}/{args.cnn}_{args.idx}.png")