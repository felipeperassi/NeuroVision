import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from sklearn.model_selection import train_test_split

from config import DEVICE, SEED, DATA_VOXELS, WEIGHTS_CLIP, WEIGHTS_VAE, RESULTS_DIR
from models import VoxelToCLIP, VoxelToVAE, VoxelToVGG

def latents_to_pil(latents, pipe):
    latents = latents / 0.18215
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
    return Image.fromarray((image * 255).astype(np.uint8))

def main():
    print("Starting NeuroVision...")
    TEST_IDX = 0  

    # Brain Data
    print("Reading brain signal...")
    X_voxels = np.load(DATA_VOXELS)
    _, X_test = train_test_split(X_voxels, test_size=0.1, random_state=SEED) # Idem to training split
    brain_signal = X_test[TEST_IDX]
    brain_tensor = torch.tensor(brain_signal).float().unsqueeze(0).to(DEVICE)

    # Load Models
    print("Loading Neural Networks...")
    
    mlp_visual = VoxelToCLIP(input_dim=X_test.shape[1]).to(DEVICE) # MLP: Voxels -> CLIP Embeddings
    mlp_visual.load_state_dict(torch.load(WEIGHTS_CLIP, map_location=DEVICE))
    mlp_visual.eval()

    cnn_vae = VoxelToVAE(input_dim=X_test.shape[1]).to(DEVICE) # CNN: Voxels -> VAE Latents
    cnn_vae.load_state_dict(torch.load(WEIGHTS_VAE, map_location=DEVICE))
    cnn_vae.eval()

    # Stable Diffusion Pipeline
    print("Initializing Stable Diffusion...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
        safety_checker=None
    ).to(DEVICE)

    # Inference Pipeline
    print("Processing brain activity...")
    with torch.no_grad():
        # Voxels -> CLIP Embeddings
        visual_embed = mlp_visual(brain_tensor)
        
        # Voxels -> VAE Latents -> PIL Image
        initial_latents = cnn_vae(brain_tensor)
        structure_pil = latents_to_pil(initial_latents, pipe)
       

    # Generate Final Image
    print("Generating final image...")
    generated_image = pipe(
        prompt_embeds=visual_embed,       # Context from MLP
        image=structure_pil,             # Structure from CNN
        strength=0.75,                  
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]

    # Save
    save_path = RESULTS_DIR / f"image_idx{TEST_IDX}.png"
    generated_image.save(save_path)
    print(f"Generated image saved to: {save_path}")

if __name__ == "__main__":
    main()