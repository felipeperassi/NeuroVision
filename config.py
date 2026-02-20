from pathlib import Path
import torch

# Directories & file paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
DATA_CLIP = DATA_DIR / 'Y_clip.npy'
DATA_TXT = DATA_DIR / 'Y_txt.npy'
DATA_VAE = DATA_DIR / 'Y_vae.npy'
DATA_VOXELS = DATA_DIR / 'X_voxs_subj01.npy'

WEIGHTS_DIR = BASE_DIR / 'weights'
WEIGHTS_AE = WEIGHTS_DIR / 'Ws_Best_AE.pth'
WEIGHTS_CLIP = WEIGHTS_DIR / 'Ws_Best_MLP_Clip.pth'
WEIGHTS_TXT = WEIGHTS_DIR / 'Ws_Best_MLP_txt.pth'
WEIGHTS_VAE = WEIGHTS_DIR / 'Ws_Best_CNN_Vae.pth'

MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

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

# Seed
SEED = 42