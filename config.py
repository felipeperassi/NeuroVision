from pathlib import Path
import torch

# Directories & file paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
WEIGHTS_DIR = BASE_DIR / 'weights'
# RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

DATA_CLIP = DATA_DIR / 'Y_clip.npy'
DATA_TXT = DATA_DIR / 'Y_txt.npy'
DATA_VAE = DATA_DIR / 'Y_vae.npy'
DATA_VOXELS = DATA_DIR / 'X_voxs_subj01.npy'

# MLP_WEIGHTS = WEIGHTS_DIR / 'Latent2CLIP_Ws.pth'
# MAPPING_WEIGHTS = WEIGHTS_DIR / 'MappingNetwork_Ws.pth'
# CNN_VAE_WEIGHTS = WEIGHTS_DIR / 'Latent2VAE_CNN_Ws.pth'

# Device configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Hardware detected: GPU")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Hardware detected: Apple Silicon")
else:
    DEVICE = "cpu"
    print("Hardware detected: CPU")