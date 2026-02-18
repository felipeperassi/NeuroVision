from .ae import Roi2LatDataset, Roi2LatAutoencoder # Autoencoder ROI -> Latent Space
from .mlp_clip import Roi2ClipDataset, Roi2ClipMLP # MLP for ROI -> CLIP
from .mlp_txt import Clip2TxtDataset, Clip2TxtMLP # MLP for CLIP -> Text
from .cnn_vae import Roi2VaeDataset, Roi2VaeCNN # CNN for ROI -> VAE