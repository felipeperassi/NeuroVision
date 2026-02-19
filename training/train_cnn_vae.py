from config import DEVICE, DATA_VOXELS, DATA_VAE, WEIGHTS_DIR
from models import Voxels2VaeDataset, Voxels2VaeCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import train_test_split
import time
import tqdm

def main():
    # Load data
    print(f"Loading data for CNN...")
    X_voxels = np.load(DATA_VOXELS)
    Y_vae = np.load(DATA_VAE) 
    X_train, X_test, Y_train, Y_test = train_test_split(X_voxels, Y_vae, test_size=0.1, random_state=42)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = Voxels2VaeDataset(X_train, Y_train)
    test_dataset = Voxels2VaeDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Initialized training in {DEVICE}.")

    # Configure model, loss, optimizer, and scheduler
    model = Voxels2VaeCNN(voxel_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Loop over epochs
    epochs = 100
    best_test_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        
        # Train loop
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(batch_x)
            
            loss = criterion(prediction, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Test loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)

        epoch_time = time.time() - start_time
        scheduler.step(avg_test_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"------------------------------------"
              f"Epoch [{epoch+1:03d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = WEIGHTS_DIR / 'Ws_Best_CNN_Vae.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved model: {save_path.name}")

    print("Finished training.")


if __name__ == "__main__":
    main()