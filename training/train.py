import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

from config import DEVICE, SEED, DATA_VOXELS, DATA_VAE, DATA_CLIP, DATA_TXT, WEIGHTS_DIR
from models import (
    Voxels2LatDataset, Voxels2LatAutoencoder, # AE
    Voxels2VaeDataset, Voxels2VaeCNN,         # CNN
    Voxels2ClipDataset, Voxels2ClipMLP,       # MLP CLIP
    Clip2TxtDataset, Clip2TxtMLP              # MLP TXT
)

# Personalized loss functions
def combined_loss_fn(pred, target, lambda_cos=1.0):
    loss_mse = F.mse_loss(pred, target)
    sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss_cos = 1.0 - sim
    return loss_mse + (lambda_cos * loss_cos)

def mapping_loss(pred, target, lambda_cos=1.0):
    loss_mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss_cos = 1.0 - cos_sim
    total_loss = loss_mse + (lambda_cos * loss_cos)
    return total_loss, cos_sim

# Configuration for each model type
CONFIGS = {
    'ae': {
        'print_name': 'Autoencoder',
        'data_in': DATA_VOXELS,
        'data_out': None, # AE is self-supervised
        'dataset': Voxels2LatDataset,
        'model': Voxels2LatAutoencoder,
        'loss': nn.MSELoss(),
        'lr': 1e-3,
        'epochs': 100,
        'save_name': 'Ws_Best_AE.pth',
        'init_args': lambda x_shape: {'input_dim': x_shape[1]}
    },
    'cnn': {
        'print_name': 'CNN',
        'data_in': DATA_VOXELS,
        'data_out': DATA_VAE,
        'dataset': Voxels2VaeDataset,
        'model': Voxels2VaeCNN,
        'loss': nn.SmoothL1Loss(),
        'lr': 1e-3,
        'epochs': 100,
        'save_name': 'Ws_Best_CNN_Vae.pth',
        'init_args': lambda x_shape: {'voxel_dim': x_shape[1]}
    },
    'mlp_clip': {
        'print_name': 'MLP Clip',
        'data_in': DATA_VOXELS,
        'data_out': DATA_CLIP,
        'dataset': Voxels2ClipDataset,
        'model': Voxels2ClipMLP,
        'loss': combined_loss_fn,
        'lr': 5e-4,
        'epochs': 60,
        'save_name': 'Ws_Best_MLP_Clip.pth',
        'init_args': lambda x_shape: {'input_dim': x_shape[1]}
    },
    'mlp_txt': {
        'print_name': 'MLP Text',
        'data_in': DATA_CLIP,
        'data_out': DATA_TXT,
        'dataset': Clip2TxtDataset,
        'model': Clip2TxtMLP,
        'loss': mapping_loss, 
        'lr': 1e-4,
        'epochs': 60,
        'save_name': 'Ws_Best_MLP_txt.pth',
        'init_args': lambda x_shape: {'input_dim': 768, 'seq_len': 77}
    }
}

# Main loop
def train(mode):
    cfg = CONFIGS[mode]
    
    # Load data
    print(f"Loading data for {cfg['print_name']}...")
    X = np.load(cfg['data_in'])
    
    # Self-Supervised vs Supervised Learning
    if cfg['data_out'] is None:
        X_train, X_test = train_test_split(X, test_size=0.1, random_state=SEED)
        Y_train, Y_test = X_train, X_test
        print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    else:
        Y = np.load(cfg['data_out'])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=SEED)
        print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = cfg['dataset'](X_train, Y_train)
    test_dataset = cfg['dataset'](X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Initialized training in {DEVICE}.")

    # Configure model, loss, optimizer, and scheduler
    model_args = cfg['init_args'](X_train.shape)
    model = cfg['model'](**model_args).to(DEVICE)
    
    criterion = cfg['loss']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Loop over epochs
    epochs = cfg['epochs']
    best_test_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        
        # Train loop
        model.train()
        train_loss = 0.0
        train_cos_acc = 0.0 # For mlp_txt

        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(batch_x)
            
            loss_val = criterion(prediction, batch_y)
            
            current_cos = 0.0
            if isinstance(loss_val, tuple):
                loss_final, current_cos = loss_val
                loss_val = loss_final
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_val.item()
            train_cos_acc += current_cos if isinstance(current_cos, float) else current_cos.item()
            
            loop.set_postfix(loss=loss_val.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_cos = train_cos_acc / len(train_loader)

        # Test loop
        model.eval()
        test_loss = 0.0
        test_cos_acc = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                prediction = model(batch_x)
                loss_val = criterion(prediction, batch_y)
                
                current_cos = 0.0
                if isinstance(loss_val, tuple):
                    loss_final, current_cos = loss_val
                    loss_val = loss_final
                
                test_loss += loss_val.item()
                test_cos_acc += current_cos if isinstance(current_cos, float) else current_cos.item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_cos = test_cos_acc / len(test_loader)

        epoch_time = time.time() - start_time
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print("------------------------------------")
        
        if mode == 'mlp_txt':
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Train Cos: {avg_train_cos:.4f} | "
                  f"Test Cos: {avg_test_cos:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Test Loss: {avg_test_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = WEIGHTS_DIR / cfg['save_name']
            torch.save(model.state_dict(), save_path)
            print(f"Saved model: {save_path.name}")

    print("Finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master training script for all models.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['ae', 'cnn', 'mlp_clip', 'mlp_txt'],
                        help='Select model: ae, cnn, mlp_clip, mlp_txt')
    
    args = parser.parse_args()
    train(args.model)