from config import (
    DATA_VOXELS, DATA_VOXELS_ST, IDX_TEST, ST_VAE, ST_VGG,
    WEIGHTS_CLIP, WEIGHTS_VAE, WEIGHTS_VGG, DIR_RESULTS, DEVICE
)
from load_data import load_inference_data

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
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
        vae_st = np.load(ST_VAE)
        mean_vae, std_vae= torch.tensor(vae_st['mean'], dtype=torch.float32).to(DEVICE), torch.tensor(vae_st['std'], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            vae_pred_norm = cnn(voxel_idx)
            vae_pred      = vae_pred_norm * std_vae.view(1, 4, 64, 64) \
                           + mean_vae.view(1, 4, 64, 64)
            vae_pred      = vae_pred / 0.18215
            decoded       = model.decode(vae_pred).sample
            decoded       = (decoded.clamp(-1, 1) + 1) / 2
            init_img = Image.fromarray(
            (decoded.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).resize((512, 512))

            return init_img

    else:
        vgg_st = np.load(ST_VGG)
        mean_vgg, std_vgg= torch.tensor(vgg_st['mean'], dtype=torch.float32).to(DEVICE), torch.tensor(vgg_st['std'], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            f1_pred, _, _ = cnn(voxel_idx)
            f1_denorm = f1_pred * std_vgg + mean_vgg
            f1_avg    = f1_denorm.mean(dim=1, keepdim=True)
            depth_map = F.interpolate(f1_avg, (512, 512), mode='bilinear', align_corners=False)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            # Convertir a RGB (ControlNet depth espera RGB)
            depth_np  = (depth_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
            depth_rgb = Image.fromarray(depth_np).convert('RGB')

            return depth_rgb

def encode_image_clip(img_pil : Image.Image) -> torch.Tensor:
    """Encodes a PIL image into CLIP embeddings using the CLIP model and processor.
        Args:
            - img_pil: A PIL Image object to be encoded into CLIP embeddings.
        Returns:
            - A normalized CLIP embedding vector representing the input image, obtained by processing the image through the CLIP model and normalizing the resulting embeddings.
    """
    clip_eval = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ).to(DEVICE).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    inputs = clip_processor(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_eval(**inputs).image_embeds
    return F.normalize(emb, dim=-1).squeeze(0)

def generate_N_best(pipe : StableDiffusionPipeline, cnn_name : str, structured_img : Image.Image, image_embeds : torch.Tensor, clip_pred : torch.Tensor, N=16) -> Image.Image:
    """
    Generates multiple candidate images using the Stable Diffusion pipeline and selects the best one based on CLIP similarity scores.
        Args:
            - pipe: The Stable Diffusion pipeline to be used for image generation.
            - cnn_name: The name of the CNN model ("VAE" or "VGG") to determine the generation parameters.
            - structured_img: The input image generated from the CNN model to be used as a conditioning image for the Stable Diffusion pipeline.
            - image_embeds: The CLIP embeddings derived from the voxel data, used for conditioning the Stable Diffusion generation.
            - clip_pred: The original CLIP prediction from the MLP model, used for scoring the generated images.
            - N: The number of candidate images to generate and evaluate (default is 16).
        Returns:
            - The generated image with the highest CLIP similarity score to the original CLIP prediction, selected from the N candidates generated by the Stable Diffusion pipeline.
    """
    candidates, scores = [], []
    clip_norm = F.normalize(clip_pred.float(), dim=-1)
    for _ in range(N):
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
                controlnet_conditioning_scale=0.35
            ).images[0]
        
        embedding = encode_image_clip(output)
        score = (embedding * clip_norm).sum().item()
        candidates.append(output)
        scores.append(score)

    return candidates[np.argmax(scores)]

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
    output = generate_N_best(pipe, cnn_name, structured_img, image_embeds, clip_pred)

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