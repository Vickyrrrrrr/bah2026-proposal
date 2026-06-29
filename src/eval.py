import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from PIL import Image

# Local imports
from dataset import SEN12MS_LISS4_SimulationDataset
from model import LISS4ClearNet

def parse_args():
    parser = argparse.ArgumentParser(description="LISS-4 ClearNet Evaluation Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SEN12MS-CR dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained model .pt checkpoint")
    parser.add_argument("--num_res_blocks", type=int, default=16, help="Number of residual blocks in model")
    parser.add_argument("--channel_dim", type=int, default=128, help="Internal channel dimension")
    parser.add_argument("--out_img_dir", type=str, default="./results", help="Directory to save visual comparison PNGs")
    return parser.parse_args()

def compute_sam(pred, gt):
    """
    Spectral Angle Mapper (SAM) in degrees.
    Computes angle between spectral vectors at each pixel. Lower is better.
    """
    # pred, gt shape: (C, H, W)
    dot = np.sum(pred * gt, axis=0)
    norm_pred = np.linalg.norm(pred, axis=0)
    norm_gt = np.linalg.norm(gt, axis=0)
    cos_val = dot / (norm_pred * norm_gt + 1e-8)
    angle_rad = np.arccos(np.clip(cos_val, -1.0, 1.0))
    return np.degrees(angle_rad).mean()

def compute_ndvi_corr(pred, gt):
    """
    Computes the Pearson correlation coefficient between predicted and target NDVI maps.
    Band indices: 0=Green, 1=Red, 2=NIR
    """
    ndvi_pred = (pred[2] - pred[1]) / (pred[2] + pred[1] + 1e-8)
    ndvi_gt = (gt[2] - gt[1]) / (gt[2] + gt[1] + 1e-8)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(ndvi_pred.flatten(), ndvi_gt.flatten())[0, 1]
    return corr if not np.isnan(corr) else 0.0

def save_rgb_false_color(tensor_chw, path):
    """
    Saves a 3-band array (Green, Red, NIR) as a standard False Color Infrared (FCIR) PNG.
    Map bands for display: NIR -> Red channel, Red -> Green channel, Green -> Blue channel.
    This is standard in remote sensing to visualize vegetation health.
    """
    arr = tensor_chw.numpy() if hasattr(tensor_chw, 'numpy') else tensor_chw
    
    # Band order: G=0, R=1, NIR=2 -> False color: NIR, R, G
    fcir = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    fcir = np.clip(fcir, 0.0, 1.0)
    img_uint8 = (fcir * 255).astype(np.uint8)
    
    Image.fromarray(img_uint8).save(path)

def evaluate(args):
    os.makedirs(args.out_img_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Evaluation running on: {device}")
    
    # Load dataset test split
    test_ds = SEN12MS_LISS4_SimulationDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Load Model
    model = LISS4ClearNet(num_res_blocks=args.num_res_blocks, channel_dim=args.channel_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print(f"✅ Loaded checkpoint: {args.checkpoint}")
    
    metrics = {"PSNR": [], "SSIM": [], "SAM": [], "NDVI_corr": []}
    
    with torch.no_grad():
        for i, (sar, cloudy, clear) in enumerate(test_loader):
            sar, cloudy = sar.to(device), cloudy.to(device)
            pred = model(cloudy, sar).cpu().squeeze(0).numpy() # (3, H, W)
            gt = clear.squeeze(0).numpy()                      # (3, H, W)
            
            # Compute evaluation metrics
            p = psnr_fn(gt, pred, data_range=1.0)
            s = ssim_fn(gt, pred, data_range=1.0, channel_axis=0)
            sam = compute_sam(pred, gt)
            ndvi_corr = compute_ndvi_corr(pred, gt)
            
            metrics["PSNR"].append(p)
            metrics["SSIM"].append(s)
            metrics["SAM"].append(sam)
            metrics["NDVI_corr"].append(ndvi_corr)
            
            # Save first 3 visual pairs for website slider update
            if i < 3:
                save_rgb_false_color(cloudy.cpu().squeeze(0), os.path.join(args.out_img_dir, f"cloudy_{i+1}.png"))
                save_rgb_false_color(pred, os.path.join(args.out_img_dir, f"clear_{i+1}.png"))
                save_rgb_false_color(gt, os.path.join(args.out_img_dir, f"gt_{i+1}.png"))
                print(f"  💾 Saved visual demonstration pair {i+1}")
                
    # Calculate means
    print("\n================ EVALUATION SUMMARY ================")
    for k, v in metrics.items():
        print(f"  {k:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    print("====================================================")
    
if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
