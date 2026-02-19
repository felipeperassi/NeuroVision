from config import DEVICE, DATA_VOXELS, DATA_CLIP, DATA_TXT, WEIGHTS_DIR
from models import Clip2TxtDataset, Clip2TxtMLP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import train_test_split
import time
import tqdm


def mapping_loss(pred, target, lambda_cos=1.0):
    loss_mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss_cos = 1.0 - cos_sim
    total_loss = loss_mse + (lambda_cos * loss_cos)
    return total_loss, cos_sim

def main():
    # Load data
    print(f"Loading data for MLP...")
    X_clip_img = np.load(DATA_CLIP)
    Y_clip_txt = np.load(DATA_TXT)
    X_train, X_test, Y_train, Y_test = train_test_split(X_clip_img, Y_clip_txt, test_size=0.1, random_state=42)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = Clip2TxtDataset(X_train, Y_train)
    test_dataset = Clip2TxtDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Initialized training in {DEVICE}.")
    
    # Configure model, loss, optimizer, and scheduler
    model = Clip2TxtMLP(input_dim=768, seq_len=77).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Loop over epochs
    epochs = 60 
    best_test_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        
        # Train loop
        model.train()
        train_loss_acc = 0.0
        train_cos_acc = 0.0
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(batch_x)
            
            loss, cos_sim = mapping_loss(prediction, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += loss.item()
            train_cos_acc += cos_sim.item()
            
            loop.set_postfix(cos=cos_sim.item())

        avg_train_loss = train_loss_acc / len(train_loader)
        avg_train_cos = train_cos_acc / len(train_loader)

        # Test loop
        model.eval()
        test_loss_acc = 0.0
        test_cos_acc = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                prediction = model(batch_x)
                loss, cos_sim = mapping_loss(prediction, batch_y)
                
                test_loss_acc += loss.item()
                test_cos_acc += cos_sim.item()
        
        avg_test_loss = test_loss_acc / len(test_loader)
        avg_test_cos = test_cos_acc / len(test_loader)

        epoch_time = time.time() - start_time
        scheduler.step(avg_test_loss)

        print(f"------------------------------------"
              f"Epoch [{epoch+1:03d}/{epochs}] | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Train Cos: {avg_train_cos:.4f} | "
              f"Test Cos: {avg_test_cos:.4f} | " 
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = WEIGHTS_DIR / 'Ws_Best_MLP_txt.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved model: {save_path.name}")


    print("Finished training.")


if __name__ == "__main__":
    main()