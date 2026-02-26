from pathlib import Path
import torch

# Directories & file paths
BASE_DIR = Path(__file__).resolve().parent

DIR_DATA    = BASE_DIR / 'data'
DIR_WEIGHTS = BASE_DIR / 'weights'
DIR_MODELS  = BASE_DIR / 'models'
DIR_RESULTS = BASE_DIR / 'results'
DIR_IDX     = BASE_DIR / 'idx'

DATA_VOXELS = DIR_DATA / 'X_VOXS.npy'
DATA_CLIP   = DIR_DATA / 'Y_CLIP.npy'
DATA_VAE    = DIR_DATA / 'Y_VAE.npy'
DATA_VGG1   = DIR_DATA / 'Y_VGG1.npy'
DATA_VGG2   = DIR_DATA / 'Y_VGG2.npy'
DATA_VGG3   = DIR_DATA / 'Y_VGG3.npy'

WEIGHTS_AE      = DIR_WEIGHTS / 'Ws_AE.pth'
WEIGHTS_CLIP    = DIR_WEIGHTS / 'Ws_MLP_CLIP.pth'
WEIGHTS_VGG     = DIR_WEIGHTS / 'Ws_MLP_VGG.pth'
WEIGHTS_VAE     = DIR_WEIGHTS / 'Ws_CNN_VAE.pth'

IDX_TRAIN = DIR_IDX / 'IDX_TRAIN.npy'
IDX_TEST  = DIR_IDX / 'IDX_TEST.npy'

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

# Seed & image idx
SEED = 42
IMG_IDX = 2