# Libraries
import matplotlib.pyplot as plt
import numpy as np

import os
import torch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
# WEIGHTS_DIR = BASE_DIR / 'weights'
# RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

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

# Data for training & testing
CLIP_IMAGE_EMBEDS = DATA_DIR / 'Y_subj01_CLIP.npy'
CLIP_TEXT_EMBEDS = DATA_DIR / 'Y_subj01_TEXT.npy'
VAE_LATENTS_PATH = DATA_DIR / 'Y_subj01_VAE.npy'

# Pesos de los Modelos (.pth)
# MLP_WEIGHTS = WEIGHTS_DIR / 'Latent2CLIP_Ws.pth'
# MAPPING_WEIGHTS = WEIGHTS_DIR / 'MappingNetwork_Ws.pth'
# CNN_VAE_WEIGHTS = WEIGHTS_DIR / 'Latent2VAE_CNN_Ws.pth'