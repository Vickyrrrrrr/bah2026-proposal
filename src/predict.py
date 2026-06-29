import os
import argparse
import numpy as np
import torch
import rasterio
from PIL import Image

# Local imports
from model import LISS4ClearNet

def parse_args():
    parser = argparse.ArgumentParser(description="LISS-4 ClearNet Single Image Cloud Removal Inference")
    parser.add_argument("--cloudy_optical", type=str, required=True, help="Path to cloudy LISS-4 GeoTIFF (Green, Red, NIR)")
    parser.add_argument("--sar_radar", type=str, required=True, help="Path to Sentinel-1 SAR GeoTIFF (VV, VH)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--output_tif", type=str, default="./results/output_clear.tif", help="Path to save georeferenced GeoTIFF output")
    parser.add_argument("--output_png", type=str, default="./results/output_clear.png", help="Path to save visual false-color PNG preview")
    parser.add_argument("--num_res_blocks", type=int, default=4, help="Number of residual blocks (must match checkpoint)")
    parser.add_argument("--channel_dim", type=int, default=64, help="Channel dimension (must match checkpoint)")
    return parser.parse_args()

def save_rgb_false_color(tensor_chw, path):
    """
    Saves a 3-band array (Green, Red, NIR) as a standard False Color Infrared (FCIR) PNG.
    Map bands for display: NIR -> Red channel, Red -> Green channel, Green -> Blue channel.
    """
    arr = tensor_chw.numpy() if hasattr(tensor_chw, 'numpy') else tensor_chw
    # Band order: G=0, R=1, NIR=2 -> False color: NIR, R, G
    fcir = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    fcir = np.clip(fcir, 0.0, 1.0)
    img_uint8 = (fcir * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    print(f"🖼️ Saved visual preview to: {path}")

def run_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running single-image inference on: {device}")
    
    # 1. Read input datasets using rasterio
    print(f"📖 Reading cloudy LISS-4: {args.cloudy_optical}")
    with rasterio.open(args.cloudy_optical) as src_opt:
        opt_meta = src_opt.meta.copy()
        # Read Green, Red, NIR (bands 1, 2, 3)
        opt_data = src_opt.read([1, 2, 3])
        opt_transform = src_opt.transform
        
    print(f"📖 Reading Sentinel-1 SAR: {args.sar_radar}")
    with rasterio.open(args.sar_radar) as src_sar:
        sar_data = src_sar.read([1, 2])
        
    # Crop to 256x256 from the center of each image to prevent CUDA Out Of Memory
    print("✂️ Cropping images to center 256x256 patch to avoid GPU OOM...")
    
    # Optical Center Crop
    c_opt, h_opt, w_opt = opt_data.shape
    start_h_opt = max(0, (h_opt - 256) // 2)
    start_w_opt = max(0, (w_opt - 256) // 2)
    opt_data = opt_data[:, start_h_opt:start_h_opt+256, start_w_opt:start_w_opt+256]
    
    # Update GeoTIFF transform for the cropped bounds
    from rasterio.windows import Window
    window = Window(start_w_opt, start_h_opt, 256, 256)
    cropped_transform = rasterio.windows.transform(window, opt_transform)
    opt_meta.update({
        "height": 256,
        "width": 256,
        "transform": cropped_transform
    })
    
    # SAR Center Crop
    c_sar, h_sar, w_sar = sar_data.shape
    start_h_sar = max(0, (h_sar - 256) // 2)
    start_w_sar = max(0, (w_sar - 256) // 2)
    sar_data = sar_data[:, start_h_sar:start_h_sar+256, start_w_sar:start_w_sar+256]
    
    # 2. Preprocess / Normalize data
    # Standard normalization matching dataset.py
    opt_data = opt_data.astype(np.float32) / 10000.0  # Assumes 16-bit DN values normalized to [0,1]
    sar_data = (sar_data.astype(np.float32) + 25.0) / 25.0  # Normalize backscatter dB
    
    # Clip and clip limits
    opt_data = np.clip(opt_data, 0.0, 1.0)
    sar_data = np.clip(sar_data, 0.0, 1.0)
    
    # Ensure correct shapes: [C, H, W] -> [1, C, H, W]
    cloudy_tensor = torch.from_numpy(opt_data).unsqueeze(0).to(device)
    sar_tensor = torch.from_numpy(sar_data).unsqueeze(0).to(device)
    
    # 3. Load Model
    model = LISS4ClearNet(num_res_blocks=args.num_res_blocks, channel_dim=args.channel_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 4. Predict
    print("⚡ Running model forward pass...")
    with torch.no_grad():
        pred_tensor = model(cloudy_tensor, sar_tensor)
        # Denormalize output to original reflectance scale for GeoTIFF
        pred_np = pred_tensor.cpu().squeeze(0).numpy()
        pred_scaled = (pred_np * 10000.0).astype(opt_meta['dtype'])
        
    # 5. Write Georeferenced GeoTIFF
    os.makedirs(os.path.dirname(args.output_tif), exist_ok=True)
    opt_meta.update({
        "count": 3,
        "driver": "GTiff"
    })
    
    print(f"💾 Writing georeferenced output: {args.output_tif}")
    with rasterio.open(args.output_tif, "w", **opt_meta) as dst:
        for b in range(3):
            dst.write(pred_scaled[b], b + 1)
            
    # 6. Save visual PNG false color preview
    if args.output_png:
        save_rgb_false_color(pred_tensor.cpu().squeeze(0), args.output_png)
        
    print("✅ Inference complete!")

if __name__ == "__main__":
    run_inference()
