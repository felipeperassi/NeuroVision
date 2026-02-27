from config import (
    DATA_VOXELS, DATA_CLIP, DATA_VAE, DATA_VGG1, DATA_VGG2, DATA_VGG3, 
    WEIGHTS_CLIP, WEIGHTS_VAE, WEIGHTS_VGG, DEVICE, PATIENCE, EPOCHS
)
from load_data import load_training_data
from models import (
    VoxelToCLIP,    # MLP for CLIP features
    VoxelToVGG,     # CNN for VGG features
    VoxelToVAE      # CNN for VAE features
)

import torch
import torch.nn.functional as F
import numpy as np
import argparse

# --------- Auxiliary functions ---------

# Mixup to augment training data
def mixup(x : torch.Tensor, y : torch.Tensor, alpha : float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Mixup augmentation to the input data.
        Args:
            - x (torch.Tensor): Input voxel data of shape (batch_size, voxel_dim).
            - y (torch.Tensor): Target CLIP embeddings of shape (batch_size, clip_dim).
            - alpha (float): Mixup interpolation coefficient (default: 0.5).
        Returns:
            - tuple[torch.Tensor, torch.Tensor]: Mixed input and target tensors.
    """
    beta  = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0), device=x.device)
    x_mix = beta * x + (1 - beta) * x[idx]
    y_mix = beta * y + (1 - beta) * y[idx]
    return x_mix, y_mix

# Loss function for CLIP training
def cos_loss(pred : torch.Tensor, gt : torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity loss between predicted and ground truth embeddings.
        Args:
            - pred (torch.Tensor): Predicted CLIP embeddings of shape (batch_size, clip_dim).
            - gt (torch.Tensor): Ground truth CLIP embeddings of shape (batch_size, clip_dim).
        Returns:
            - torch.Tensor: Combined loss value.
    """
    pred = F.normalize(pred, dim=-1)
    gt   = F.normalize(gt, dim=-1)
    cosine = (1 - (pred * gt).sum(dim=-1)).mean()
    return cosine

# --------- Main training function ---------

# Configuration for each model type
MODELS = {
    'MLP' : {
        'model' : VoxelToCLIP,
        'weights': WEIGHTS_CLIP,
        't1': DATA_CLIP,
        't2': None,
        't3': None,
        'target_name': "CLIP",
        'loss_fn': cos_loss,
        'lr': 1e-4,
        'batch_size': 256,
        'wd': 1e-4
    },

    'CNN_VAE' : {
        'model' : VoxelToVAE,
        'weights': WEIGHTS_VAE,
        't1': DATA_VAE,
        't2': None,
        't3': None,
        'target_name': "VAE",
        'loss_fn': F.mse_loss,
        'lr': 1e-4,
        'batch_size': 32,
        'wd': 1e-3,
    },

    'CNN_VGG' : {
        'model' : VoxelToVGG,
        'weights': WEIGHTS_VGG,
        't1': DATA_VGG1,
        't2': DATA_VGG2,
        't3': DATA_VGG3,
        'target_name': "VGG",
        'loss_fn': F.mse_loss,
        'lr': 1e-4,
        'batch_size': 128,
        'wd': 1e-4
    }
} 

# Main training loop
def train(mode : str) -> None:
    """
    Trains the specified model type (MLP, CNN_VAE, or CNN_VGG) using the corresponding dataset and configuration.
        Args:
            - mode (str): The model type to train ('MLP', 'CNN_VAE', or 'CNN_VGG').
    """
    # Check for valid mode and get corresponding configuration
    if mode not in MODELS:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(MODELS.keys())}")
    model = MODELS[mode]
    
    # Load data
    print(f"Loading data for {model['target_name']}...")
    train_loader, test_loader, mean, std = load_training_data(
        voxels_path=DATA_VOXELS, t1_path=model['t1'], t2_path=model['t2'], t3_path=model['t3'], 
        target_name=model['target_name'], batch_size=model['batch_size']
    )
    print(f"Initialized training in {DEVICE}.")

    # Configure model, loss, optimizer, and scheduler
    input_dim   = next(iter(train_loader))[0].shape[1]
    train_model = model['model'](input_dim=input_dim).to(DEVICE)
    optimizer   = torch.optim.AdamW(train_model.parameters(), lr=model['lr'], weight_decay=model['wd'])
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # General training loop
    for epoch in range(EPOCHS):
        
        # Training phase
        train_model.train()
        train_loss = 0.0

        if model['target_name'] == "CLIP" or model['target_name'] == "VAE":
            for voxels, gt in train_loader:
                voxels, gt = voxels.to(DEVICE), gt.to(DEVICE)
                if model['target_name'] == "CLIP": voxels, gt = mixup(voxels, gt, alpha=0.6)
                    
                pred = train_model(voxels)
                loss = model['loss_fn'](pred, gt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

        else: # VGG case
            for voxels, gt1, gt2, gt3 in train_loader:
                voxels = voxels.to(DEVICE)
                gt1, gt2, gt3 = gt1.to(DEVICE), gt2.to(DEVICE), gt3.to(DEVICE)
                pred1, pred2, pred3 = train_model(voxels)

                loss = model['loss_fn'](pred1, gt1) \
                    + model['loss_fn'](pred2, gt2) \
                    + model['loss_fn'](pred3, gt3)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

        # Validation phase
        train_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            if model['target_name'] == "CLIP" or model['target_name'] == "VAE":
                for voxels, gt in test_loader:
                    voxels, gt = voxels.to(DEVICE), gt.to(DEVICE)
                    pred = train_model(voxels)
                    val_loss += model['loss_fn'](pred, gt).item()
            
            else: # VGG case
                for voxels, gt1, gt2, gt3 in test_loader:
                    voxels = voxels.to(DEVICE)
                    gt1, gt2, gt3 = gt1.to(DEVICE), gt2.to(DEVICE), gt3.to(DEVICE)
                    pred1, pred2, pred3 = train_model(voxels)
                    val_loss    += (model['loss_fn'](pred1, gt1)
                                + model['loss_fn'](pred2, gt2)
                                + model['loss_fn'](pred3, gt3)).item()

        train_loss /= len(train_loader)
        val_loss   /= len(test_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        # Save best model and implement early stopping
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": train_model.state_dict(),
                "input_dim":   input_dim
            }, model['weights'])
            print(f"--------- Saved ---------")
        
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nFinished training, best validation loss: {best_val_loss:.4f}. Model saved to {model['weights']}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master training script for all models.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['MLP', 'CNN_VAE', 'CNN_VGG'], help='Select model: MLP, CNN_VAE, CNN_VGG')
    
    args = parser.parse_args()
    train(args.model)