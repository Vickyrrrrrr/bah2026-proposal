import sys
from pathlib import Path

# Add src to python path
src_path = Path(__file__).resolve().parent
sys.path.append(str(src_path))

try:
    print("Testing imports...")
    import numpy as np
    import rasterio
    import skimage
    import requests
    import shapely
    
    from bhoonidhi_client import BhoonidhiClient
    from preprocess import RemoteSensingPreprocessor
    
    print("[SUCCESS] Core GIS/downloader imports successful!")
    
    try:
        import torch
        from dataset import SEN12MS_LISS4_SimulationDataset, LISS4RealDataset
        from model import LISS4ClearNet
        from train import ndvi_preservation_loss
        from eval import compute_sam, compute_ndvi_corr
        
        print("[SUCCESS] PyTorch/DL imports successful!")
        
        print("Testing model initialization...")
        model = LISS4ClearNet(num_res_blocks=4, channel_dim=32)
        opt = torch.randn(1, 3, 128, 128)
        sar = torch.randn(1, 2, 128, 128)
        out = model(opt, sar)
        print(f"[SUCCESS] Model successfully ran forward pass! Output shape: {out.shape}")
    except ModuleNotFoundError:
        print("[WARNING] PyTorch not installed locally. DL training scripts (train.py, eval.py, dataset.py, model.py) are prepared and should be run in Google Colab.")

except Exception as e:
    print(f"[ERROR] Error during dry-run verification: {e}")
    sys.exit(1)
