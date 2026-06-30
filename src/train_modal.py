import os
import sys
import modal

# Define the Modal App
app = modal.App("liss4-clearnet-training")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "rasterio",
        "scikit-image",
        "tqdm",
        "requests",
        "wandb"
    )
    .add_local_dir("./src", remote_path="/root/src")
)

# Persistent volume to store dataset and save model checkpoints
volume = modal.Volume.from_name("liss4-clearnet-volume", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4",  # L4 is highly cost-effective (24GB VRAM, GDDR6, cheaper than A100/A10G)
    volumes={"/workspace": volume},
    timeout=7200  # 2 hours maximum run time
)
def run_training_remote(epochs: int = 60, batch_size: int = 4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import numpy as np

    # Add mount path to system path for imports
    sys.path.append("/root/src")

    workspace_path = "/workspace"

    # 2. Local imports from the mount
    from dataset import SEN12MS_LISS4_SimulationDataset
    from model import LISS4ClearNet
    from train import ndvi_preservation_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Remote Modal GPU Active: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Paths inside the persistent volume
    data_dir = os.path.join(workspace_path, "data")
    checkpoints_dir = os.path.join(workspace_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Check if dataset exists in the volume, if not, download it
    dataset_root = os.path.join(data_dir, "ROIs1158_spring")
    if not os.path.exists(dataset_root) or not os.listdir(dataset_root):
        print("⏳ Dataset not found in persistent volume. Downloading dataset split...")
        os.makedirs(dataset_root, exist_ok=True)
        # Import and run stream extraction helper directly
        from download_helper import stream_extract_url
        urls = [
            "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s1.tar.gz",
            "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_cloudy.tar.gz",
            "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"
        ]
        for url in urls:
            stream_extract_url(url, dataset_root, max_roi=30)

    # Initialize Datasets and Loaders
    print("⏳ Loading dataset partitions...")
    train_ds = SEN12MS_LISS4_SimulationDataset(data_dir, split="train")
    val_ds = SEN12MS_LISS4_SimulationDataset(data_dir, split="val")
    
    # L4 GPU has 24GB VRAM, allowing larger batch size and multiple workers
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Instantiate full 16-block ClearNet model
    print("🤖 Instantiating full LISS-4 ClearNet (16 Blocks, 128 Channels)...")
    model = LISS4ClearNet(num_res_blocks=16, channel_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_psnr = -float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sar, cloudy, clear) in enumerate(train_loader):
            sar, cloudy, clear = sar.to(device), cloudy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            pred = model(cloudy, sar)
            
            loss_pixel = F.mse_loss(pred, clear)
            loss_ndvi = ndvi_preservation_loss(pred, clear)
            loss = loss_pixel + 0.5 * loss_ndvi
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_idx:3d}/{len(train_loader)} | Loss: {loss.item():.5f}")
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_mse = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for sar, cloudy, clear in val_loader:
                sar, cloudy, clear = sar.to(device), cloudy.to(device), clear.to(device)
                pred = model(cloudy, sar)
                
                batch_mse = F.mse_loss(pred, clear, reduction="sum").item()
                val_mse += batch_mse
                n_val_samples += sar.shape[0] * 3 * sar.shape[2] * sar.shape[3]
                
        val_mse = val_mse / n_val_samples
        val_psnr = 10 * np.log10(1.0 / (val_mse + 1e-8))
        
        print(f"🔔 Epoch {epoch:02d}/{epochs} Complete | Train Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            checkpoint_path = os.path.join(checkpoints_dir, "best_clearnet.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Saved best checkpoint to: {checkpoint_path} (PSNR: {val_psnr:.2f} dB)")
            # Commit files back to the persistent volume
            volume.commit()
            
    print(f"🎉 Modal Remote Training Complete! Best Validation PSNR: {best_val_psnr:.2f} dB")

@app.local_entrypoint()
def main(epochs: int = 60, batch_size: int = 16):
    print("🚀 Triggering LISS-4 ClearNet remote training on Modal cloud...")
    # First, copy local src directory files into the persistent volume workspace
    # This ensures the remote task has the latest codebase files
    print("📂 Syncing local src codebase to persistent Modal volume...")
    
    # Trigger the remote training run
    run_training_remote.remote(epochs=epochs, batch_size=batch_size)
