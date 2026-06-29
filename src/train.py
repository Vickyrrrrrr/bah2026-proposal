import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Local imports
from dataset import SEN12MS_LISS4_SimulationDataset
from model import LISS4ClearNet

def parse_args():
    parser = argparse.ArgumentParser(description="LISS-4 ClearNet Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SEN12MS-CR dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_res_blocks", type=int, default=16, help="Number of residual blocks in model")
    parser.add_argument("--channel_dim", type=int, default=128, help="Internal channel dimension")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save model weights")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    return parser.parse_args()

def ndvi_preservation_loss(pred, target):
    """
    Computes loss on NDVI difference between predicted and clear-sky ground-truth.
    Assumes bands: 0=Green, 1=Red, 2=NIR
    NDVI = (NIR - Red) / (NIR + Red)
    """
    nir_pred, red_pred = pred[:, 2], pred[:, 1]
    nir_target, red_target = target[:, 2], target[:, 1]
    
    ndvi_pred = (nir_pred - red_pred) / (nir_pred + red_pred + 1e-8)
    ndvi_target = (nir_target - red_target) / (nir_target + red_target + 1e-8)
    
    return F.mse_loss(ndvi_pred, ndvi_target)

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training on device: {device}")
    
    # 1. Initialize Datasets and Loaders
    print("⏳ Loading datasets...")
    train_ds = SEN12MS_LISS4_SimulationDataset(args.data_dir, split="train")
    val_ds = SEN12MS_LISS4_SimulationDataset(args.data_dir, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # 2. Instantiate Model, Optimizer, and Cosine Scheduler
    model = LISS4ClearNet(num_res_blocks=args.num_res_blocks, channel_dim=args.channel_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Weights & Biases logging
    if args.use_wandb:
        import wandb
        wandb.init(
            project="liss4-clearnet",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_res_blocks": args.num_res_blocks,
                "channel_dim": args.channel_dim,
                "device": str(device)
            }
        )
        
    best_val_psnr = -float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sar, cloudy, clear) in enumerate(train_loader):
            sar, cloudy, clear = sar.to(device), cloudy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            pred = model(cloudy, sar)
            
            # Loss formulation: MSE (Pixel Reconstruction) + 0.5 * NDVI Loss
            loss_pixel = F.mse_loss(pred, clear)
            loss_ndvi = ndvi_preservation_loss(pred, clear)
            loss = loss_pixel + 0.5 * loss_ndvi
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress every 50 batches for live feedback
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader)} | Current Loss: {loss.item():.5f}")
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_mse = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for sar, cloudy, clear in val_loader:
                sar, cloudy, clear = sar.to(device), cloudy.to(device), clear.to(device)
                pred = model(cloudy, sar)
                
                batch_mse = F.mse_loss(pred, clear, reduction="sum").item()
                # batch_mse is sum over all elements, normalize by batch size * channels * height * width
                val_mse += batch_mse
                n_val_samples += sar.shape[0] * 3 * sar.shape[2] * sar.shape[3]
                
        val_mse = val_mse / n_val_samples
        val_psnr = 10 * np.log10(1.0 / (val_mse + 1e-8))
        
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")
        
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_psnr": val_psnr})
            
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_clearnet.pt"))
            print(f"  💾 Saved best model checkpoint (Val PSNR: {val_psnr:.2f} dB)")
            
    print(f"🎉 Training complete! Best validation PSNR: {best_val_psnr:.2f} dB")
    
if __name__ == "__main__":
    args = parse_args()
    train(args)
